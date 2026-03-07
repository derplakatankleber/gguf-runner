use super::{VendorDecodePolicy, VendorMultimodalPolicy, VendorRuntimeDebugPolicy};
use crate::engine::types::{
    Tokenizer, VendorTokenizerPolicy, LLAMA3_BOS_TOKEN, LLAMA3_END_HEADER, LLAMA3_EOT,
    LLAMA3_START_HEADER,
};

pub(super) fn default_rope_theta() -> f32 {
    500_000.0
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: &[],
        deterministic_loop_guard: false,
    }
}

pub(super) fn tokenizer_policy() -> VendorTokenizerPolicy {
    VendorTokenizerPolicy::default()
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy::default()
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    VendorRuntimeDebugPolicy::default()
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos = tokenizer
        .find_special_token("<|begin_of_text|>")
        .unwrap_or(LLAMA3_BOS_TOKEN);
    let start_header = tokenizer
        .find_special_token("<|start_header_id|>")
        .unwrap_or(LLAMA3_START_HEADER);
    let end_header = tokenizer
        .find_special_token("<|end_header_id|>")
        .unwrap_or(LLAMA3_END_HEADER);
    let eot = tokenizer
        .find_special_token("<|eot_id|>")
        .unwrap_or(LLAMA3_EOT);

    tokens.push(bos);

    if !system_prompt.is_empty() {
        tokens.push(start_header);
        tokenizer.bpe_encode("system", &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_header);
        tokenizer.bpe_encode(&format!("\n\n{}", system_prompt), &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(eot);
    }

    tokens.push(start_header);
    tokenizer.bpe_encode("user", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode(&format!("\n\n{}", prompt), &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(eot);

    tokens.push(start_header);
    tokenizer.bpe_encode("assistant", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode("\n\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}
