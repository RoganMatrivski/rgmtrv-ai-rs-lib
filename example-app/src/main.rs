use rgmt_core::{Tool, async_trait};
use rgmt_state::AppState;
use rgmt_vllm::VllmInstance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 1. Define a custom result type for our app.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct MathAssistantResult {
    explanation: String,
    final_answer: f64,
}

/// 2. Define a custom tool by implementing the `Tool` trait.
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic math operations"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn call(&self, args: serde_json::Value) -> eyre::Result<String> {
        let op = args["operation"].as_str().unwrap();
        let a = args["a"].as_f64().unwrap();
        let b = args["b"].as_f64().unwrap();

        let res = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => a / b,
            _ => unreachable!(),
        };

        Ok(res.to_string())
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    println!("--- Tool-Enabled Generic LLM Framework Example ---");

    let instance = VllmInstance::new(
        "https://llama-cpp.rgmtrv.my.id/v1",
        "unsloth/Qwen3.5-2B-GGUF",
        "no-key-needed",
    );

    // 3. Register tools with the AppState.
    let app =
        AppState::<MathAssistantResult>::new(instance, None).add_tool(Arc::new(CalculatorTool));

    let system_prompt = r#"You are a math assistant.
Use the calculator tool for any calculations.
Return a JSON object with:
- "explanation": string
- "final_answer": number"#;

    let user_prompt = "What is (123 + 456) * 7.8?";

    println!("Running state machine with tools...");

    // The state machine will:
    // 1. Send the request to LLM.
    // 2. See that LLM wants to call 'calculator'.
    // 3. Call our CalculatorTool::call.
    // 4. Send the result back to LLM.
    // 5. Finally parse the MathAssistantResult.
    let result = app.run(None, system_prompt, user_prompt).await;

    match result {
        Ok(res) => {
            println!("\nResult: {:#?}", res);
        }
        Err(e) => {
            eprintln!("\nError (Expected if no local LLM): {:?}", e);
        }
    }

    Ok(())
}
