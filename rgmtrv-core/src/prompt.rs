pub const IMAGE_INFERENCE_PROMPT: &str = "Describe this image.";

pub const REPAIR_PROMPT: &str = r#"/no_think
The previous response was invalid. 
Message: {msg}
Errors:
{err}
Context:
{data}
Please fix the output and return only the corrected JSON object."#;
