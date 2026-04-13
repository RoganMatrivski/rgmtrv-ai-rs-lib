use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, ChatCompletionTool, ChatCompletionTools,
        CreateChatCompletionRequestArgs, ImageUrl,
    },
};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use eyre::WrapErr;
use futures::StreamExt;
use image::GenericImageView;

use rgmt_imgproc::ImageProcessor;

// -- Pending parts ------------------------------------------------------------

#[derive(Clone)]
enum PendingPart {
    Text(String),
    Image(image::DynamicImage),
}

#[derive(Clone)]
struct PendingMessage {
    role: String,
    parts: Vec<PendingPart>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
    tool_call_id: Option<String>,
}

// -- MessageStack -------------------------------------------------------------

#[derive(Clone)]
pub struct MessageStack {
    messages: Vec<PendingMessage>,
}

impl MessageStack {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.messages.push(PendingMessage {
            role: "system".into(),
            parts: vec![PendingPart::Text(text.into())],
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.messages.push(PendingMessage {
            role: "user".into(),
            parts: vec![PendingPart::Text(text.into())],
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    pub fn user_image_text(mut self, img: image::DynamicImage, text: impl Into<String>) -> Self {
        self.messages.push(PendingMessage {
            role: "user".into(),
            parts: vec![PendingPart::Image(img), PendingPart::Text(text.into())],
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    pub fn user_image(mut self, img: image::DynamicImage) -> Self {
        self.messages.push(PendingMessage {
            role: "user".into(),
            parts: vec![PendingPart::Image(img)],
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    pub fn assistant_text(mut self, text: impl Into<String>) -> Self {
        self.messages.push(PendingMessage {
            role: "assistant".into(),
            parts: vec![PendingPart::Text(text.into())],
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    pub fn assistant_tool_calls(
        mut self,
        text: Option<String>,
        tool_calls: Vec<ChatCompletionMessageToolCall>,
    ) -> Self {
        self.messages.push(PendingMessage {
            role: "assistant".into(),
            parts: text.map(|t| vec![PendingPart::Text(t)]).unwrap_or_default(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        });
        self
    }

    pub fn tool_response(mut self, id: impl Into<String>, content: impl Into<String>) -> Self {
        self.messages.push(PendingMessage {
            role: "tool".into(),
            parts: vec![PendingPart::Text(content.into())],
            tool_calls: None,
            tool_call_id: Some(id.into()),
        });
        self
    }

    pub fn resolve(self) -> eyre::Result<Vec<ChatCompletionRequestMessage>> {
        let mut out = Vec::with_capacity(self.messages.len());

        for pending in self.messages {
            let message = match pending.role.as_str() {
                "system" => {
                    let text = collect_text(pending.parts);
                    ChatCompletionRequestMessage::System(
                        ChatCompletionRequestSystemMessageArgs::default()
                            .content(text)
                            .build()?,
                    )
                }
                "assistant" => {
                    let text = collect_text(pending.parts);
                    let mut args = ChatCompletionRequestAssistantMessageArgs::default();
                    if !text.is_empty() {
                        args.content(text);
                    }
                    if let Some(tool_calls) = pending.tool_calls {
                        let tool_calls: Vec<ChatCompletionMessageToolCalls> = tool_calls
                            .into_iter()
                            .map(ChatCompletionMessageToolCalls::Function)
                            .collect();
                        args.tool_calls(tool_calls);
                    }
                    ChatCompletionRequestMessage::Assistant(args.build()?)
                }
                "tool" => {
                    let text = collect_text(pending.parts);
                    ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(pending.tool_call_id.unwrap_or_default())
                            .content(text)
                            .build()?,
                    )
                }
                _ => {
                    let mut content_parts: Vec<ChatCompletionRequestUserMessageContentPart> =
                        Vec::with_capacity(pending.parts.len());

                    for part in pending.parts {
                        match part {
                            PendingPart::Text(text) => {
                                content_parts.push(
                                    ChatCompletionRequestMessageContentPartText { text }.into(),
                                );
                            }
                            PendingPart::Image(img) => {
                                let img = img_process(img)?;
                                let url = img_to_data_url(img)?;
                                content_parts.push(
                                    ChatCompletionRequestMessageContentPartImage {
                                        image_url: ImageUrl {
                                            url,
                                            detail: Some(
                                                async_openai::types::chat::ImageDetail::Auto,
                                            ),
                                        },
                                    }
                                    .into(),
                                );
                            }
                        }
                    }

                    ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(ChatCompletionRequestUserMessageContent::Array(
                                content_parts,
                            ))
                            .build()?,
                    )
                }
            };

            out.push(message);
        }

        Ok(out)
    }
}

impl Default for MessageStack {
    fn default() -> Self {
        Self::new()
    }
}

// -- ChatResponse -------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

// -- ChatBuilder --------------------------------------------------------------

pub struct ChatBuilder<'a> {
    instance: &'a VllmInstance,
    stack: MessageStack,
    tools: Vec<ChatCompletionTool>,
}

impl<'a> ChatBuilder<'a> {
    fn new(instance: &'a VllmInstance) -> Self {
        Self {
            instance,
            stack: MessageStack::new(),
            tools: Vec::new(),
        }
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.stack = self.stack.system(text);
        self
    }

    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.stack = self.stack.user_text(text);
        self
    }

    pub fn user_image_text(mut self, img: image::DynamicImage, text: impl Into<String>) -> Self {
        self.stack = self.stack.user_image_text(img, text);
        self
    }

    pub fn user_image(mut self, img: image::DynamicImage) -> Self {
        self.stack = self.stack.user_image(img);
        self
    }

    pub fn assistant_text(mut self, text: impl Into<String>) -> Self {
        self.stack = self.stack.assistant_text(text);
        self
    }

    pub fn tools(mut self, tools: Vec<ChatCompletionTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a function tool to the request.
    pub fn add_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        self.tools.push(ChatCompletionTool {
            function: async_openai::types::chat::FunctionObject {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
                strict: Some(true),
            },
        });
        self
    }

    pub async fn send(self) -> eyre::Result<String> {
        let messages = self.stack.resolve()?;
        let res = self
            .instance
            .stream_messages(messages, Some(self.tools))
            .await?;
        Ok(res.content.unwrap_or_default())
    }

    pub async fn send_full(self) -> eyre::Result<ChatResponse> {
        let messages = self.stack.resolve()?;
        self.instance
            .stream_messages(messages, Some(self.tools))
            .await
    }

    pub fn get_stack(&self) -> MessageStack {
        self.stack.clone()
    }
}

// -- VllmInstance -------------------------------------------------------------

#[derive(Clone)]
pub struct VllmInstance {
    client: Client<OpenAIConfig>,
    pub model: String,
}

impl VllmInstance {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(base_url.into())
            .with_api_key(api_key.into());

        Self {
            client: Client::with_config(config),
            model: model.into(),
        }
    }

    pub fn chat(&self) -> ChatBuilder<'_> {
        ChatBuilder::new(self)
    }

    pub async fn stream_messages(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Option<Vec<ChatCompletionTool>>,
    ) -> eyre::Result<ChatResponse> {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.model)
            .messages(messages)
            .temperature(0.0f32)
            .top_p(1.0f32)
            .max_completion_tokens(8192u32)
            .stream(true)
            .frequency_penalty(0.1f32);

        if let Some(tools) = tools {
            if !tools.is_empty() {
                let tools: Vec<ChatCompletionTools> = tools
                    .into_iter()
                    .map(ChatCompletionTools::Function)
                    .collect();
                builder.tools(tools);
            }
        }

        let request = builder
            .build()
            .wrap_err("Failed to build chat completion request")?;

        let mut stream = self
            .client
            .chat()
            .create_stream(request)
            .await
            .wrap_err("Failed to open vLLM stream")?;

        let mut output = String::new();
        let mut tool_calls: Vec<ChatCompletionMessageToolCall> = Vec::new();

        while let Some(result) = stream.next().await {
            let response = result?;
            for choice in response.choices {
                if let Some(content) = choice.delta.content {
                    output.push_str(&content);
                }

                if let Some(delta_tool_calls) = choice.delta.tool_calls {
                    for chunk in delta_tool_calls {
                        let idx = chunk.index as usize;
                        if tool_calls.len() <= idx {
                            tool_calls.push(ChatCompletionMessageToolCall {
                                id: chunk.id.clone().unwrap_or_default(),
                                function: async_openai::types::chat::FunctionCall {
                                    name: chunk
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.name.clone())
                                        .unwrap_or_default(),
                                    arguments: chunk
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.clone())
                                        .unwrap_or_default(),
                                },
                            });
                        } else {
                            if let Some(id) = chunk.id {
                                tool_calls[idx].id.push_str(&id);
                            }
                            if let Some(f) = chunk.function {
                                if let Some(name) = f.name {
                                    tool_calls[idx].function.name.push_str(&name);
                                }
                                if let Some(args) = f.arguments {
                                    tool_calls[idx].function.arguments.push_str(&args);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ChatResponse {
            content: if output.is_empty() {
                None
            } else {
                Some(output)
            },
            tool_calls,
        })
    }
}

// -- Helpers ------------------------------------------------------------------

fn img_to_data_url(img: image::DynamicImage) -> eyre::Result<String> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
    Ok(format!("data:image/jpeg;base64,{}", B64.encode(&buf)))
}

fn collect_text(parts: Vec<PendingPart>) -> String {
    parts
        .into_iter()
        .filter_map(|p| match p {
            PendingPart::Text(t) => Some(t),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn img_process(img: image::DynamicImage) -> eyre::Result<image::DynamicImage> {
    let proc = ImageProcessor::builder()
        .image(img)
        .build()
        .resize_longest_side(1024);

    let img = proc.process();
    let (w, h) = img.dimensions();

    let new_w = ((w + 15) / 16) * 16;
    let new_h = ((h + 15) / 16) * 16;

    let img = ImageProcessor::builder()
        .image(img)
        .build()
        .pad_right(new_w - w)
        .pad_bottom(new_h - h)
        .process();

    Ok(img)
}
