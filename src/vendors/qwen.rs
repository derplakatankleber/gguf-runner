use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GenerationRequest, MediaRef, PlaceholderSpan, ThinkMode,
    Tokenizer,
};

pub(super) fn finalize_moe_config(config: &mut Config) -> Result<(), String> {
    if config.expert_hidden_dim == 0 || config.n_experts == 0 {
        return Err(
            "qwen model is missing expert metadata (expert_count/expert_feed_forward_length)"
                .to_string(),
        );
    }

    if config.n_experts_used == 0 {
        config.n_experts_used = 1;
    }
    if config.n_experts_used > config.n_experts {
        config.n_experts_used = config.n_experts;
    }

    Ok(())
}

pub(super) fn apply_qwen3moe_defaults(config: &mut Config) {
    // Qwen3 MoE routing defaults from official config.json.
    config.moe_n_group = 8;
    config.moe_topk_group = 4;
    config.moe_norm_topk_prob = true;
    config.moe_routed_scaling_factor = 1.0;
}

pub(super) fn validate_qwen3next(config: &mut Config) -> Result<(), String> {
    // Qwen3Next uses normalized top-k expert weights with softmax gating.
    config.moe_norm_topk_prob = true;
    config.moe_routed_scaling_factor = 1.0;

    if config.ssm_conv_kernel == 0
        || config.ssm_inner_size == 0
        || config.ssm_state_size == 0
        || config.ssm_time_step_rank == 0
        || config.ssm_group_count == 0
    {
        return Err(
            "qwen3next model is missing SSM metadata (ssm.conv_kernel/inner_size/state_size/time_step_rank/group_count)"
                .to_string(),
        );
    }

    if !config
        .ssm_inner_size
        .is_multiple_of(config.ssm_time_step_rank)
    {
        return Err(format!(
            "qwen3next invalid SSM metadata: inner_size {} not divisible by time_step_rank {}",
            config.ssm_inner_size, config.ssm_time_step_rank
        ));
    }
    if !config
        .ssm_time_step_rank
        .is_multiple_of(config.ssm_group_count)
    {
        return Err(format!(
            "qwen3next invalid SSM metadata: time_step_rank {} not divisible by group_count {}",
            config.ssm_time_step_rank, config.ssm_group_count
        ));
    }

    let head_v_dim = config.ssm_inner_size / config.ssm_time_step_rank;
    if head_v_dim != config.ssm_state_size {
        return Err(format!(
            "qwen3next unsupported SSM shape: state_size {} != inner_size/time_step_rank {}",
            config.ssm_state_size, head_v_dim
        ));
    }

    Ok(())
}

pub(super) fn print_qwen3moe_debug(config: &Config) {
    eprintln!(
        "Qwen3MoE: experts={}, experts_used={}, n_group={}, topk_group={}, norm_topk_prob={}, routed_scaling_factor={}, rms_norm_eps={}",
        config.n_experts,
        config.n_experts_used,
        config.moe_n_group,
        config.moe_topk_group,
        config.moe_norm_topk_prob,
        config.moe_routed_scaling_factor,
        config.rms_norm_eps
    );
}

pub(super) fn print_qwen3next_debug(config: &Config) {
    eprintln!(
        "Qwen3Next: experts={}, experts_used={}, expert_hidden_dim={}, shared_expert_hidden_dim={}, ssm_inner={}, ssm_state={}, ssm_heads={}, ssm_groups={}, ssm_conv_kernel={}, rms_norm_eps={}",
        config.n_experts,
        config.n_experts_used,
        config.expert_hidden_dim,
        config.shared_expert_hidden_dim,
        config.ssm_inner_size,
        config.ssm_state_size,
        config.ssm_time_step_rank,
        config.ssm_group_count,
        config.ssm_conv_kernel,
        config.rms_norm_eps
    );
}

pub(super) fn encode_qwen2_chat(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = if system_prompt.is_empty() {
        "You are a helpful assistant."
    } else {
        system_prompt
    };

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        tokens.push(start);
        tokenizer.bpe_encode("system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        return tokens;
    }

    // Fallback: encode ChatML markers as plain text if special tokens are not mapped.
    let rendered = format!(
        "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}

pub(super) fn encode_qwen3_chat(
    tokenizer: &mut Tokenizer,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    let mut parts = Vec::with_capacity(1 + image_count);
    parts.push(ContentPart::Text(prompt.to_string()));
    for _ in 0..image_count {
        parts.push(ContentPart::Image(MediaRef {
            path: String::new(),
        }));
    }
    let request = GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
    };
    encode_qwen3_request(tokenizer, &request, think_mode).token_ids
}

fn append_encoded_literal(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    literal: &str,
) -> (usize, usize) {
    let start = tokens.len();
    tokenizer.bpe_encode(literal, temp);
    tokens.extend_from_slice(temp);
    (start, tokens.len().saturating_sub(start))
}

fn append_vision_wrapped_placeholder(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    vision_start: Option<i32>,
    pad_token: Option<i32>,
    vision_end: Option<i32>,
    pad_literal: &str,
) -> (usize, usize) {
    if let (Some(vs), Some(pad), Some(ve)) = (vision_start, pad_token, vision_end) {
        let start = tokens.len();
        tokens.push(vs);
        tokens.push(pad);
        tokens.push(ve);
        (start, 3)
    } else {
        let rendered = format!("<|vision_start|>{pad_literal}<|vision_end|>");
        append_encoded_literal(tokenizer, temp, tokens, &rendered)
    }
}

fn append_audio_placeholder(
    tokenizer: &mut Tokenizer,
    temp: &mut Vec<i32>,
    tokens: &mut Vec<i32>,
    vision_start: Option<i32>,
    audio_pad: Option<i32>,
    vision_end: Option<i32>,
) -> (usize, usize) {
    if let (Some(vs), Some(ap), Some(ve)) = (vision_start, audio_pad, vision_end) {
        let start = tokens.len();
        tokens.push(vs);
        tokens.push(ap);
        tokens.push(ve);
        return (start, 3);
    }
    if let Some(ap) = audio_pad {
        let start = tokens.len();
        tokens.push(ap);
        return (start, 1);
    }
    append_encoded_literal(tokenizer, temp, tokens, "<|audio_pad|>")
}

pub(super) fn encode_qwen3_request(
    tokenizer: &mut Tokenizer,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let mut image_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut video_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut audio_spans: Vec<PlaceholderSpan> = Vec::new();
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut audio_index = 0usize;
    let sys = request.system_prompt.trim();

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");
    let vision_start = tokenizer.find_special_token("<|vision_start|>");
    let vision_end = tokenizer.find_special_token("<|vision_end|>");
    let image_pad = tokenizer.find_special_token("<|image_pad|>");
    let video_pad = tokenizer.find_special_token("<|video_pad|>");
    let audio_pad = tokenizer.find_special_token("<|audio_pad|>");

    if tokenizer.bos_token >= 0 {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        if !sys.is_empty() {
            tokens.push(start);
            tokenizer.bpe_encode("system\n", &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode(sys, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        for part in &request.parts {
            match part {
                ContentPart::Text(text) => {
                    tokenizer.bpe_encode(text, &mut temp);
                    tokens.extend_from_slice(&temp);
                }
                ContentPart::Image(_) => {
                    let (token_start, token_len) = append_vision_wrapped_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        vision_start,
                        image_pad,
                        vision_end,
                        "<|image_pad|>",
                    );
                    image_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: image_index,
                    });
                    image_index += 1;
                    tokenizer.bpe_encode("\n", &mut temp);
                    tokens.extend_from_slice(&temp);
                }
                ContentPart::Video(_) => {
                    let (token_start, token_len) = append_vision_wrapped_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        vision_start,
                        video_pad,
                        vision_end,
                        "<|video_pad|>",
                    );
                    video_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: video_index,
                    });
                    video_index += 1;
                    tokenizer.bpe_encode("\n", &mut temp);
                    tokens.extend_from_slice(&temp);
                }
                ContentPart::Audio(_) => {
                    let (token_start, token_len) = append_audio_placeholder(
                        tokenizer,
                        &mut temp,
                        &mut tokens,
                        vision_start,
                        audio_pad,
                        vision_end,
                    );
                    audio_spans.push(PlaceholderSpan {
                        token_start,
                        token_len,
                        media_index: audio_index,
                    });
                    audio_index += 1;
                    tokenizer.bpe_encode("\n", &mut temp);
                    tokens.extend_from_slice(&temp);
                }
            }
        }
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("<think>\n", &mut temp);
        tokens.extend_from_slice(&temp);

        return EncodedPrompt {
            token_ids: tokens,
            image_spans,
            video_spans,
            audio_spans,
        };
    }

    if !sys.is_empty() {
        tokenizer.bpe_encode("<|im_start|>system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode("<|im_end|>\n", &mut temp);
        tokens.extend_from_slice(&temp);
    }

    tokenizer.bpe_encode("<|im_start|>user\n", &mut temp);
    tokens.extend_from_slice(&temp);
    for part in &request.parts {
        match part {
            ContentPart::Text(text) => {
                tokenizer.bpe_encode(text, &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Image(_) => {
                let (token_start, token_len) = append_vision_wrapped_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    vision_start,
                    image_pad,
                    vision_end,
                    "<|image_pad|>",
                );
                image_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: image_index,
                });
                image_index += 1;
                tokenizer.bpe_encode("\n", &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Video(_) => {
                let (token_start, token_len) = append_vision_wrapped_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    vision_start,
                    video_pad,
                    vision_end,
                    "<|video_pad|>",
                );
                video_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: video_index,
                });
                video_index += 1;
                tokenizer.bpe_encode("\n", &mut temp);
                tokens.extend_from_slice(&temp);
            }
            ContentPart::Audio(_) => {
                let (token_start, token_len) = append_audio_placeholder(
                    tokenizer,
                    &mut temp,
                    &mut tokens,
                    vision_start,
                    audio_pad,
                    vision_end,
                );
                audio_spans.push(PlaceholderSpan {
                    token_start,
                    token_len,
                    media_index: audio_index,
                });
                audio_index += 1;
                tokenizer.bpe_encode("\n", &mut temp);
                tokens.extend_from_slice(&temp);
            }
        }
    }
    // Choose the assistant turn opening based on think mode:
    // - Yes/Hidden: open <think> block (model will generate thinking then </think> then answer)
    // - No: immediately close <think></think> so the model skips thinking
    let assistant_suffix = if think_mode == ThinkMode::No {
        "<|im_end|>\n<|im_start|>assistant\n<think></think>\n"
    } else {
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    };
    tokenizer.bpe_encode(assistant_suffix, &mut temp);
    tokens.extend_from_slice(&temp);

    EncodedPrompt {
        token_ids: tokens,
        image_spans,
        video_spans,
        audio_spans,
    }
}

#[cfg(test)]
mod tests {
    use super::encode_qwen3_request;
    use crate::engine::types::{ContentPart, GenerationRequest, MediaRef, ThinkMode, Tokenizer};

    fn tokenizer_with_qwen_specials() -> Tokenizer {
        Tokenizer {
            vocab: vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<|vision_start|>".to_string(),
                "<|vision_end|>".to_string(),
                "<|image_pad|>".to_string(),
                "<|video_pad|>".to_string(),
                "<|audio_pad|>".to_string(),
            ],
            ..Tokenizer::default()
        }
    }

    #[test]
    fn qwen3_request_maps_multimodal_placeholder_spans() {
        let mut tokenizer = tokenizer_with_qwen_specials();
        let request = GenerationRequest {
            system_prompt: String::new(),
            parts: vec![
                ContentPart::Text("analyze".to_string()),
                ContentPart::Image(MediaRef {
                    path: "a.png".to_string(),
                }),
                ContentPart::Video(MediaRef {
                    path: "b.mp4".to_string(),
                }),
                ContentPart::Audio(MediaRef {
                    path: "c.wav".to_string(),
                }),
            ],
        };
        let encoded = encode_qwen3_request(&mut tokenizer, &request, ThinkMode::Yes);

        assert_eq!(encoded.image_spans.len(), 1);
        assert_eq!(encoded.video_spans.len(), 1);
        assert_eq!(encoded.audio_spans.len(), 1);

        let vision_start = tokenizer
            .find_special_token("<|vision_start|>")
            .expect("vision_start");
        let vision_end = tokenizer
            .find_special_token("<|vision_end|>")
            .expect("vision_end");
        let image_pad = tokenizer
            .find_special_token("<|image_pad|>")
            .expect("image_pad");
        let video_pad = tokenizer
            .find_special_token("<|video_pad|>")
            .expect("video_pad");
        let audio_pad = tokenizer
            .find_special_token("<|audio_pad|>")
            .expect("audio_pad");

        let image_span = encoded.image_spans[0];
        assert_eq!(image_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [image_span.token_start..image_span.token_start + image_span.token_len],
            &[vision_start, image_pad, vision_end]
        );

        let video_span = encoded.video_spans[0];
        assert_eq!(video_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [video_span.token_start..video_span.token_start + video_span.token_len],
            &[vision_start, video_pad, vision_end]
        );

        let audio_span = encoded.audio_spans[0];
        assert_eq!(audio_span.token_len, 3);
        assert_eq!(
            &encoded.token_ids
                [audio_span.token_start..audio_span.token_start + audio_span.token_len],
            &[vision_start, audio_pad, vision_end]
        );
    }
}
