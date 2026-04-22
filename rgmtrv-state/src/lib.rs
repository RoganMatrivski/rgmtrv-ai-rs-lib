#[cfg(feature = "image")]
use rgmtrv_core::img_process;
use rgmtrv_core::{Tool, extract_json, strip_fences, try_parse_json};
use rgmtrv_vllm::{MessageStack, OpenAiInstance};
use std::sync::Arc;
use strum_macros::Display;

// ── Domain Types ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum RepairOrigin {
    /// Came from State::Search  → return via ReturnSearch
    FromSearch,
    /// Came from State::Finalize → return via Save
    FromFinalize,
}

#[derive(Debug)]
pub struct RepairPayload<T> {
    pub message: String,
    /// All validation/parse errors collected in this repair cycle.
    pub causes: Vec<eyre::Report>,
    pub data: RepairData,
    pub origin: RepairOrigin,
    pub _marker: std::marker::PhantomData<T>,
}

#[derive(Debug)]
pub enum RepairData {
    /// Raw JSON string that failed to parse — passed verbatim to the repair prompt.
    RawJson(String),
    /// Original tool-call payload whose arguments failed validation.
    ToolCall(
        (
            Option<String>,
            Vec<async_openai::types::chat::ChatCompletionMessageToolCall>,
        ),
    ),
}

#[derive(Debug, serde::Deserialize)]
pub struct JsonToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, serde::Deserialize)]
pub struct JsonToolCallWrapper {
    pub tool_call: JsonToolCall,
}

type ToolCall = (
    Option<String>,
    Vec<async_openai::types::chat::ChatCompletionMessageToolCall>,
);

#[derive(Debug, Display)]
pub enum State<T> {
    Init,
    Infer,
    Search(ToolCall),
    ReturnSearch(ToolCall),
    Finalize(String),
    Save(T),
    Repair(RepairPayload<T>),
    Exit,
}

impl<T> State<T> {
    pub fn progress_step(&self) -> u64 {
        match self {
            State::Init => 0,
            State::Infer => 1,
            State::Search(_) | State::ReturnSearch(_) => 2,
            State::Repair(_) => 2,
            State::Finalize(_) => 3,
            State::Save(_) => 4,
            State::Exit => 5,
        }
    }

    pub fn progress_label(&self) -> &'static str {
        match self {
            State::Init => "Initializing...",
            State::Infer => "Inferring...",
            State::Search(_) => "Searching...",
            State::ReturnSearch(_) => "Processing tool results...",
            State::Repair(_) => "! Repairing...",
            State::Finalize(_) => "Finalizing...",
            State::Save(_) => "Saving...",
            State::Exit => "Done",
        }
    }

    fn variant_name(&self) -> &'static str {
        match self {
            State::Init => "Init",
            State::Infer => "Infer",
            State::Search(_) => "Search",
            State::ReturnSearch(_) => "ReturnSearch",
            State::Finalize(_) => "Finalize",
            State::Save(_) => "Save",
            State::Repair(_) => "Repair",
            State::Exit => "Exit",
        }
    }
}

// ── State Machine Core ───────────────────────────────────────────────────────

pub struct StateMachine<T> {
    pub current: State<T>,
}

impl<T> StateMachine<T> {
    pub fn new() -> Self {
        Self {
            current: State::Init,
        }
    }

    /// Transition to `next`, validating the edge is legal.
    pub fn transition(&mut self, next: State<T>) -> eyre::Result<&State<T>> {
        let valid = matches!(
            (&self.current, &next),
            (State::Init, State::Infer)
                | (State::Infer, State::Search(_))
                | (State::Infer, State::Finalize(_))
                | (State::Search(_), State::ReturnSearch(_))
                | (State::Search(_), State::Repair(_))
                | (State::ReturnSearch(_), State::Search(_))
                | (State::ReturnSearch(_), State::Finalize(_))
                | (State::Finalize(_), State::Save(_))
                | (State::Finalize(_), State::Repair(_))
                | (State::Repair(_), State::Save(_))
                | (State::Repair(_), State::Repair(_))
                | (State::Repair(_), State::ReturnSearch(_))
                | (State::Save(_), State::Exit)
        );

        if !valid {
            eyre::bail!(
                "Invalid transition: {} → {}",
                self.current.variant_name(),
                next.variant_name()
            );
        }

        self.current = next;
        Ok(&self.current)
    }
}

// ── Application Logic ────────────────────────────────────────────────────────

pub struct AppState<T> {
    instance: OpenAiInstance,
    pb: Option<indicatif::ProgressBar>,
    state: StateMachine<T>,
    tools: Vec<Arc<dyn Tool>>,
    repair_prompt: String,
}

impl<T> AppState<T>
where
    T: serde::de::DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub fn new(instance: OpenAiInstance, pb: Option<indicatif::ProgressBar>) -> Self {
        Self {
            instance,
            pb,
            state: StateMachine::new(),
            tools: Vec::new(),
            repair_prompt: r#"/no_think
The previous response was invalid. 
Message: {msg}
Errors:
{err}
Context:
{data}
Please fix the output and return only the corrected JSON object."#.to_string(),
        }
    }

    pub fn with_repair_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.repair_prompt = prompt.into();
        self
    }

    pub fn add_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub async fn run(
        mut self,
        #[cfg(feature = "image")] img: Option<image::DynamicImage>,
        system_prompt: impl Into<String>,
        user_prompt: impl Into<String>,
    ) -> eyre::Result<T> {
        let builder = self.initialize_msgstack().await?;

        let mut msgstack = builder.system(system_prompt);

        #[cfg(feature = "image")]
        {
            if let Some(img) = img {
                let img = img_process(img)?;
                msgstack = msgstack.user_image_text(img, user_prompt);
            } else {
                msgstack = msgstack.user_text(user_prompt);
            }
        }
        #[cfg(not(feature = "image"))]
        {
            msgstack = msgstack.user_text(user_prompt);
        }

        let mut msgstack = msgstack.get_stack();

        self.setup_progress_bar();
        self.state.transition(State::Infer)?;

        while !matches!(self.state.current, State::Exit) {
            self.update_progress_bar();

            match &self.state.current {
                State::Infer => self.handle_infer(&mut msgstack).await?,
                State::Search(_) => self.handle_search().await?,
                State::ReturnSearch(_) => self.handle_return_search(&mut msgstack).await?,
                State::Finalize(_) => self.handle_finalize().await?,
                State::Save(result) => {
                    let res = result.clone();
                    self.state.transition(State::Exit)?;
                    if let Some(pb) = &self.pb {
                        pb.set_position(5);
                        pb.finish_and_clear();
                    }
                    return Ok(res);
                }
                State::Repair(_) => self.handle_repair(&mut msgstack).await?,
                State::Exit => break,
                State::Init => unreachable!(),
            };
        }

        eyre::bail!("State machine exited without returning a result")
    }

    async fn initialize_msgstack<'a>(&'a self) -> eyre::Result<rgmtrv_vllm::ChatBuilder<'a>> {
        let mut builder = self.instance.chat();
        for tool in &self.tools {
            builder = builder.add_tool(tool.name(), tool.description(), tool.parameters());
        }
        Ok(builder)
    }

    fn setup_progress_bar(&self) {
        if let Some(pb) = &self.pb {
            pb.set_length(5);
            pb.set_style(
                indicatif::ProgressStyle::with_template("[{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
        }
    }

    fn update_progress_bar(&self) {
        if let Some(pb) = &self.pb {
            let current = &self.state.current;
            pb.set_position(current.progress_step());
            pb.set_message(current.progress_label());
        }
    }

    // ── State Handlers ───────────────────────────────────────────────────────

    async fn handle_infer(&mut self, msgstack: &mut MessageStack) -> eyre::Result<()> {
        let res = self
            .instance
            .stream_messages(msgstack.clone().resolve()?, None, None, false)
            .await?;

        if res.tool_calls.is_empty() {
            if let Some(content) = &res.content
                && let Some(tc) = self.detect_json_tool_call(content)
            {
                self.state
                    .transition(State::Search((res.content, vec![tc])))?;
            } else if let Some(n) = res.content {
                self.state.transition(State::Finalize(n))?;
            } else {
                eyre::bail!("Infer produced no content or tool calls");
            }
        } else {
            self.state
                .transition(State::Search((res.content, res.tool_calls)))?;
        }
        Ok(())
    }

    async fn handle_search(&mut self) -> eyre::Result<()> {
        let (msg, tool_calls) = match &self.state.current {
            State::Search(tc) => tc.clone(),
            _ => unreachable!(),
        };

        let mut errors = Vec::<eyre::Report>::new();
        for tc in &tool_calls {
            if let Some(tool) = self.tools.iter().find(|t| t.name() == tc.function.name) {
                match serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                    Ok(args) => {
                        if let Err(e) = tool.validate(&args) {
                            errors.push(e);
                        }
                    }
                    Err(e) => {
                        errors.push(eyre::eyre!(
                            "Invalid JSON in tool arguments for {}: {}",
                            tc.function.name,
                            e
                        ));
                    }
                }
            } else {
                errors.push(eyre::eyre!("Tool {} not found", tc.function.name));
            }
        }

        if !errors.is_empty() {
            self.state.transition(State::Repair(RepairPayload {
                message: format!(
                    "{} tool call(s) had invalid arguments or were not found",
                    errors.len()
                ),
                causes: errors,
                data: RepairData::ToolCall((msg, tool_calls)),
                origin: RepairOrigin::FromSearch,
                _marker: std::marker::PhantomData,
            }))?;
        } else {
            self.state
                .transition(State::ReturnSearch((msg, tool_calls)))?;
        }
        Ok(())
    }

    async fn handle_return_search(&mut self, msgstack: &mut MessageStack) -> eyre::Result<()> {
        let (msg, tool_calls) = match &self.state.current {
            State::ReturnSearch(tc) => tc.clone(),
            _ => unreachable!(),
        };

        *msgstack = msgstack
            .clone()
            .assistant_tool_calls(msg.clone(), tool_calls.clone());

        for tc in &tool_calls {
            let tool = self.tools.iter().find(|t| t.name() == tc.function.name);
            let res = if let Some(tool) = tool {
                let args = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)?;
                tool.call(args).await?
            } else {
                eyre::bail!("Tool {} not found during execution", tc.function.name);
            };
            *msgstack = msgstack.clone().tool_response(tc.id.clone(), res);
        }

        let res = self
            .instance
            .stream_messages(msgstack.clone().resolve()?, None, None, false)
            .await?;

        if res.tool_calls.is_empty() {
            if let Some(content) = &res.content
                && let Some(tc) = self.detect_json_tool_call(content)
            {
                self.state
                    .transition(State::Search((res.content, vec![tc])))?;
            } else if let Some(n) = res.content {
                self.state.transition(State::Finalize(n))?;
            } else {
                eyre::bail!("ReturnSearch produced no content or tool calls");
            }
        } else {
            self.state
                .transition(State::Search((res.content, res.tool_calls)))?;
        }
        Ok(())
    }

    async fn handle_finalize(&mut self) -> eyre::Result<()> {
        let rawstr = match &self.state.current {
            State::Finalize(s) => s.clone(),
            _ => unreachable!(),
        };
        let extracted = extract_json(strip_fences(&rawstr));

        match try_parse_json::<T>(extracted) {
            Ok(result) => self.state.transition(State::Save(result))?,
            Err(e) => {
                tracing::warn!(error = %e, "JSON parse failed, attempting repair");
                self.state.transition(State::Repair(RepairPayload {
                    message: "Failed parsing JSON".into(),
                    causes: vec![e],
                    data: RepairData::RawJson(extracted.to_owned()),
                    origin: RepairOrigin::FromFinalize,
                    _marker: std::marker::PhantomData,
                }))?
            }
        };
        Ok(())
    }

    async fn handle_repair(&mut self, msgstack: &mut MessageStack) -> eyre::Result<()> {
        let (err_msg, origin) = match &self.state.current {
            State::Repair(payload) => {
                let data_context = match &payload.data {
                    RepairData::RawJson(raw) => format!("Broken JSON:\n{raw}"),
                    RepairData::ToolCall((_, tcs)) => format!(
                        "Tool calls with invalid arguments:\n{}",
                        serde_json::to_string_pretty(tcs).unwrap_or_default()
                    ),
                };
                let all_errors = payload
                    .causes
                    .iter()
                    .enumerate()
                    .map(|(i, e)| format!("{}. {e}", i + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                let err_msg = self.repair_prompt
                    .replace("{msg}", &payload.message)
                    .replace("{err}", &all_errors)
                    .replace("{data}", &data_context);
                let origin = match payload.origin {
                    RepairOrigin::FromSearch => RepairOrigin::FromSearch,
                    RepairOrigin::FromFinalize => RepairOrigin::FromFinalize,
                };
                (err_msg, origin)
            }
            _ => unreachable!(),
        };

        *msgstack = msgstack.clone().user_text(err_msg);
        let res = self
            .instance
            .stream_messages(msgstack.clone().resolve()?, None, None, false)
            .await?;

        match origin {
            RepairOrigin::FromFinalize => {
                let raw = res
                    .content
                    .ok_or_else(|| eyre::eyre!("Repair produced no content"))?;
                let extracted = extract_json(strip_fences(&raw));

                match try_parse_json::<T>(extracted) {
                    Ok(result) => {
                        tracing::debug!("Repair succeeded, transitioning to save state");
                        self.state.transition(State::Save(result))?;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Repair attempt failed, retrying");
                        self.state.transition(State::Repair(RepairPayload {
                            message: "Repair still produced invalid JSON".into(),
                            causes: vec![e],
                            data: RepairData::RawJson(raw),
                            origin: RepairOrigin::FromFinalize,
                            _marker: std::marker::PhantomData,
                        }))?;
                    }
                }
            }
            RepairOrigin::FromSearch => {
                let mut errors = Vec::<eyre::Report>::new();
                for tc in &res.tool_calls {
                    if let Some(tool) = self.tools.iter().find(|t| t.name() == tc.function.name) {
                        match serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                            Ok(args) => {
                                if let Err(e) = tool.validate(&args) {
                                    errors.push(e);
                                }
                            }
                            Err(e) => {
                                errors.push(eyre::eyre!(
                                    "Invalid JSON in tool arguments for {}: {}",
                                    tc.function.name,
                                    e
                                ));
                            }
                        }
                    } else {
                        errors.push(eyre::eyre!("Tool {} not found", tc.function.name));
                    }
                }

                if errors.is_empty() {
                    tracing::debug!("Repair succeeded, returning to search");
                    self.state
                        .transition(State::ReturnSearch((res.content, res.tool_calls)))?;
                } else {
                    tracing::warn!(count = errors.len(), "Repair attempt failed, retrying");
                    self.state.transition(State::Repair(RepairPayload {
                        message: format!(
                            "Repair still produced {} invalid tool call argument(s)",
                            errors.len()
                        ),
                        causes: errors,
                        data: RepairData::ToolCall((res.content, res.tool_calls)),
                        origin: RepairOrigin::FromSearch,
                        _marker: std::marker::PhantomData,
                    }))?;
                }
            }
        }
        Ok(())
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    fn detect_json_tool_call(
        &self,
        content: &str,
    ) -> Option<async_openai::types::chat::ChatCompletionMessageToolCall> {
        let extracted = extract_json(strip_fences(content));
        if let Ok(wrapper) = serde_json::from_str::<JsonToolCallWrapper>(extracted) {
            return Some(async_openai::types::chat::ChatCompletionMessageToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                function: async_openai::types::chat::FunctionCall {
                    name: wrapper.tool_call.name,
                    arguments: wrapper.tool_call.arguments.to_string(),
                },
            });
        }
        None
    }
}
