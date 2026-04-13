pub mod prompt;

use std::path::Path;
pub use serde::{Deserialize, Serialize};
use eyre::ContextCompat;

pub use async_trait::async_trait;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Summary {
    pub summary: String,
    pub tags: Vec<String>,
    pub texts: Vec<String>,
    pub urls: Vec<String>,
    pub notes: serde_json::Value,
    pub intent: Option<String>,
    pub category: Option<String>,
    pub footnotes: Option<String>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;

    /// Optional validation step before execution.
    fn validate(&self, _args: &serde_json::Value) -> eyre::Result<()> {
        Ok(())
    }

    async fn call(&self, args: serde_json::Value) -> eyre::Result<String>;
}

pub fn get_filename(p: impl AsRef<Path>) -> eyre::Result<String> {
    Ok(p.as_ref()
        .file_name()
        .wrap_err("Cannot get filename from a dir path")?
        .to_string_lossy()
        .to_string())
}

/// Some models wrap their JSON in ```json ... ``` — strip it before parsing.
pub fn strip_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

pub fn extract_json(raw: &str) -> &str {
    // Strip <think>...</think> block if present
    let after_think = if let Some(end) = raw.find("</think>") {
        &raw[end + "</think>".len()..]
    } else {
        raw
    };

    // Find the first { and last } to extract JSON
    let start = after_think.find('{').unwrap_or(0);
    let end = after_think
        .rfind('}')
        .map(|i| i + 1)
        .unwrap_or(after_think.len());

    after_think[start..end].trim()
}

pub fn img_process(img: image::DynamicImage) -> eyre::Result<image::DynamicImage> {
    use image::GenericImageView;
    use rgmt_imgproc::ImageProcessor;

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

pub async fn get_model(baseurl: impl AsRef<str>) -> eyre::Result<String> {
    let baseurl = url::Url::parse(baseurl.as_ref())?;
    let model_url = baseurl.join("v1/models")?;

    let client = reqwest::Client::new();
    let response = client.get(model_url).send().await?;

    let json: serde_json::Value = response.json().await?;

    Ok(json["data"][0]["id"]
        .as_str()
        .wrap_err("Cannot find default model ID")?
        .to_string())
}

pub fn fix_json_escapes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                    result.push(c);
                } else {
                    result.push('\\');
                    result.push('\\');
                }
            } else {
                result.push('\\');
                result.push('\\');
            }
        } else {
            result.push(c);
        }
    }
    result
}

pub fn try_parse_json<T>(raw: &str) -> eyre::Result<T> 
where T: serde::de::DeserializeOwned 
{
    if let Ok(s) = serde_json::from_str::<T>(raw) {
        return Ok(s);
    }
    let fixed_escapes = fix_json_escapes(raw);
    if let Ok(s) = serde_json::from_str::<T>(&fixed_escapes) {
        return Ok(s);
    }

    if let Ok(fixed) = json_fixer::JsonFixer::fix(raw) {
        if let Ok(s) = serde_json::from_str::<T>(&fixed) {
            return Ok(s);
        }
        let fixed_both = fix_json_escapes(&fixed);
        if let Ok(s) = serde_json::from_str::<T>(&fixed_both) {
            return Ok(s);
        }
    }
    serde_json::from_str::<T>(raw).map_err(eyre::Report::from)
}
