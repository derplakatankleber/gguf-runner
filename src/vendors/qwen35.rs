use super::{
    qwen_common, ChatMessage, VendorDecodePolicy, VendorDetailCropPolicy, VendorMultimodalPolicy,
    VendorRuntimeDebugPolicy, VendorTokenizerPolicy,
};
use crate::engine::types::{EncodedPrompt, GenerationRequest, ThinkMode, Tokenizer};

fn qwen35_detail_crop_enabled() -> bool {
    matches!(
        std::env::var("GGUF_QWEN35_DETAIL_CROP"),
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    )
}

pub(super) fn decode_policy() -> VendorDecodePolicy {
    VendorDecodePolicy {
        parse_think_tags: true,
        stop_token_literals: qwen_common::QWEN_STOP_TOKEN_LITERALS,
        deterministic_loop_guard: true,
        deterministic_loop_guard_min_generated_tokens: 96,
        recover_early_endoftext_once: false,
        early_endoftext_recover_max_tokens: 0,
        hidden_think_token_cap_base: 384,
        visible_think_token_cap_base: 192,
        prefer_hidden_think_for_multimodal: true,
        retry_without_think_when_no_post_think_text: true,
    }
}

pub(super) fn tokenizer_policy() -> VendorTokenizerPolicy {
    VendorTokenizerPolicy {
        disable_bos_fallback: true,
        end_turn_token_literals: qwen_common::QWEN_END_TURN_TOKEN_LITERALS,
    }
}

pub(super) fn multimodal_policy() -> VendorMultimodalPolicy {
    VendorMultimodalPolicy {
        image_prompt_suffix: "\nPlease avoid guessing uncertain details. If text is unclear, explicitly say it is unreadable.",
        detail_crop: VendorDetailCropPolicy {
            enabled: qwen35_detail_crop_enabled(),
            max_layers: 24,
            note_text: "\n(Second image: centered close-up crop of the same source.)\n",
            temp_file_prefix: "gguf-runner-qwen35-detail",
        },
        mmproj_filename_score_hints: qwen_common::QWEN_MMPROJ_SCORE_HINTS,
        missing_sidecar_hint: " hint: Qwen3.5 image/video inputs require a compatible Qwen3.5 mmproj sidecar from the same checkpoint family.",
    }
}

pub(super) fn runtime_debug_policy() -> VendorRuntimeDebugPolicy {
    qwen_common::runtime_debug_policy()
}

pub(super) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_chat(tokenizer, prompt, system_prompt, image_count, think_mode)
}

pub(super) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    qwen_common::encode_qwen3_messages(tokenizer, messages, system_prompt, think_mode)
}

pub(super) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    qwen_common::encode_qwen3_request(tokenizer, request, think_mode)
}
