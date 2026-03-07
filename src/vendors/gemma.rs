use super::{VendorDecodePolicy, VendorMultimodalPolicy, VendorRuntimeDebugPolicy};
use crate::engine::types::{
    Config, Tokenizer, VendorTokenizerPolicy, GEMMA3_BOS_TOKEN, GEMMA3_END_TURN, GEMMA3_START_TURN,
};

pub(super) fn default_rope_theta() -> f32 {
    1_000_000.0
}

pub(super) fn print_config_debug(config: &Config) {
    eprintln!(
        "Gemma3: rms_norm_eps={}, final_logit_softcapping={}",
        config.rms_norm_eps, config.final_logit_softcapping
    );
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: false,
        stop_token_literals: &["<end_of_turn>"],
        deterministic_loop_guard: false,
    }
}

pub(super) fn tokenizer_policy() -> VendorTokenizerPolicy {
    VendorTokenizerPolicy {
        disable_bos_fallback: false,
        end_turn_token_literals: &["<end_of_turn>"],
    }
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

    let bos_token = tokenizer
        .find_special_token("<bos>")
        .unwrap_or(GEMMA3_BOS_TOKEN);
    let start_turn = tokenizer
        .find_special_token("<start_of_turn>")
        .unwrap_or(GEMMA3_START_TURN);
    let end_turn = tokenizer
        .find_special_token("<end_of_turn>")
        .unwrap_or(GEMMA3_END_TURN);

    tokens.push(bos_token);

    let full_prompt = if !system_prompt.is_empty() {
        format!("{}\n\n{}", system_prompt, prompt)
    } else {
        prompt.to_string()
    };

    tokens.push(start_turn);
    tokenizer.bpe_encode(&format!("user\n{}", full_prompt), &mut temp);
    tokens.extend_from_slice(&temp);

    tokens.push(end_turn);
    tokenizer.bpe_encode("\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens.push(start_turn);
    tokenizer.bpe_encode("model\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}
