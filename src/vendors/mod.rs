mod gemma;
mod llama;
mod qwen;

use crate::engine::io::{
    get_gguf_float_from_map, get_gguf_i64_array_from_map, get_gguf_int_from_map,
    get_gguf_string_from_map,
};
use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GGUFFile, GenerationRequest, ModelCapabilities,
    MultimodalBackend, ThinkMode, Tokenizer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelFamily {
    Llama,
    Gemma,
    Qwen2,
    Qwen35,
    Qwen3Vl,
    Qwen3Moe,
    Qwen3Next,
}

struct ModelIdentity {
    key_prefix: String,
    family: ModelFamily,
}

const VISION_TENSOR_PREFIXES: &[&str] = &[
    "v.",
    "mm.",
    "vision.",
    "visual.",
    "vision_tower.",
    "multi_modal_projector.",
    "mmproj.",
];
const AUDIO_TENSOR_PREFIXES: &[&str] = &["audio.", "aud.", "speech."];

fn has_vocab_token(gguf: &GGUFFile, token: &str) -> bool {
    gguf.vocab_tokens.iter().any(|entry| entry == token)
}

fn has_tensor_with_any_prefix(gguf: &GGUFFile, prefixes: &[&str]) -> bool {
    gguf.tensors.iter().any(|tensor| {
        prefixes
            .iter()
            .any(|prefix| tensor.name.starts_with(prefix))
    })
}

fn detect_model_capabilities(
    gguf: &GGUFFile,
    family: ModelFamily,
    debug_mode: bool,
) -> ModelCapabilities {
    let multimodal_backend = match family {
        ModelFamily::Qwen3Vl => MultimodalBackend::Qwen3Vl,
        ModelFamily::Qwen35 => MultimodalBackend::Qwen35,
        _ => MultimodalBackend::None,
    };

    if multimodal_backend == MultimodalBackend::None {
        return ModelCapabilities::default();
    }

    let has_vision_start = has_vocab_token(gguf, "<|vision_start|>");
    let has_vision_end = has_vocab_token(gguf, "<|vision_end|>");
    let has_image_pad = has_vocab_token(gguf, "<|image_pad|>");
    let has_video_pad = has_vocab_token(gguf, "<|video_pad|>");
    let has_audio_pad = has_vocab_token(gguf, "<|audio_pad|>");
    let has_vision_tensors = has_tensor_with_any_prefix(gguf, VISION_TENSOR_PREFIXES);
    let has_audio_tensors = has_tensor_with_any_prefix(gguf, AUDIO_TENSOR_PREFIXES);

    let capabilities = ModelCapabilities {
        multimodal_backend,
        supports_native_image: has_vision_start
            && has_vision_end
            && has_image_pad
            && has_vision_tensors,
        supports_native_video: has_vision_start
            && has_vision_end
            && has_video_pad
            && has_vision_tensors,
        supports_native_audio: has_audio_pad && has_audio_tensors,
    };

    if debug_mode {
        eprintln!(
            "Multimodal capability probe (backend={}): tokens(image={} video={} audio={}) tensors(vision={} audio={}) native(image={} video={} audio={})",
            capabilities.multimodal_backend.as_str(),
            has_vision_start && has_vision_end && has_image_pad,
            has_vision_start && has_vision_end && has_video_pad,
            has_audio_pad,
            has_vision_tensors,
            has_audio_tensors,
            capabilities.supports_native_image,
            capabilities.supports_native_video,
            capabilities.supports_native_audio
        );
    }

    capabilities
}

fn detect_model_identity(gguf: &GGUFFile, debug_mode: bool) -> ModelIdentity {
    let arch = get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or("llama");
    if debug_mode {
        eprintln!("Model architecture: {arch}");
    }

    let mut identity = ModelIdentity {
        key_prefix: "llama".to_string(),
        family: ModelFamily::Llama,
    };

    if arch == "gemma3" || arch == "gemma2" || arch == "gemma" {
        identity.family = ModelFamily::Gemma;
        identity.key_prefix = "gemma3".to_string();
        if get_gguf_int_from_map(&gguf.kv, "gemma3.embedding_length", 0) == 0 {
            if get_gguf_int_from_map(&gguf.kv, "gemma2.embedding_length", 0) != 0 {
                identity.key_prefix = "gemma2".to_string();
            } else if get_gguf_int_from_map(&gguf.kv, "gemma.embedding_length", 0) != 0 {
                identity.key_prefix = "gemma".to_string();
            }
        }
        if debug_mode {
            eprintln!(
                "Detected Gemma architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen35" || arch.starts_with("qwen35") {
        identity.family = ModelFamily::Qwen35;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0 {
            if get_gguf_int_from_map(&gguf.kv, "qwen35.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen35".to_string();
            } else if get_gguf_int_from_map(&gguf.kv, "qwen2.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen2".to_string();
            }
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3.5 architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen3vl" || arch.starts_with("qwen3vl") {
        identity.family = ModelFamily::Qwen3Vl;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "qwen3vl.embedding_length", 0) != 0
        {
            identity.key_prefix = "qwen3vl".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3-VL architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen3moe" || arch.starts_with("qwen3moe") {
        identity.family = ModelFamily::Qwen3Moe;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "qwen3moe.embedding_length", 0) != 0
        {
            identity.key_prefix = "qwen3moe".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3 MoE architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch == "qwen3next" || arch.starts_with("qwen3next") {
        identity.family = ModelFamily::Qwen3Next;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0
            && get_gguf_int_from_map(&gguf.kv, "qwen3next.embedding_length", 0) != 0
        {
            identity.key_prefix = "qwen3next".to_string();
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen3 Next architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    } else if arch.starts_with("qwen") || arch == "qwen2" {
        identity.family = ModelFamily::Qwen2;
        identity.key_prefix = arch.to_string();
        let probe = format!("{}.embedding_length", identity.key_prefix);
        if get_gguf_int_from_map(&gguf.kv, &probe, 0) == 0 {
            if get_gguf_int_from_map(&gguf.kv, "qwen2.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen2".to_string();
            } else if get_gguf_int_from_map(&gguf.kv, "qwen.embedding_length", 0) != 0 {
                identity.key_prefix = "qwen".to_string();
            }
        }
        if debug_mode {
            eprintln!(
                "Detected Qwen architecture, using {}.* keys",
                identity.key_prefix
            );
        }
    }

    identity
}

pub(crate) fn build_config_from_gguf(gguf: &GGUFFile, debug_mode: bool) -> Result<Config, String> {
    let arch = get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or("llama");
    let is_qwen3vlmoe_arch = arch == "qwen3vlmoe" || arch.starts_with("qwen3vlmoe");
    if arch.starts_with("deepseek") {
        return Err(format!(
            "unsupported GGUF architecture '{arch}': DeepSeek models are not implemented yet in this runtime (supported: llama, gemma, qwen2, qwen35, qwen3vl, qwen3moe, qwen3next)"
        ));
    }
    let identity = detect_model_identity(gguf, debug_mode);
    let capabilities = detect_model_capabilities(gguf, identity.family, debug_mode);
    let key_prefix = identity.key_prefix;

    let key_dim = format!("{key_prefix}.embedding_length");
    let key_hidden = format!("{key_prefix}.feed_forward_length");
    let key_layers = format!("{key_prefix}.block_count");
    let key_heads = format!("{key_prefix}.attention.head_count");
    let key_kv_heads = format!("{key_prefix}.attention.head_count_kv");
    let key_vocab = format!("{key_prefix}.vocab_size");
    let key_ctx = format!("{key_prefix}.context_length");
    let key_rope = format!("{key_prefix}.rope.freq_base");
    let key_rope_dim = format!("{key_prefix}.rope.dimension_count");
    let key_rope_sections = format!("{key_prefix}.rope.dimension_sections");
    let key_head_dim = format!("{key_prefix}.attention.key_length");
    let key_n_deepstack = format!("{key_prefix}.n_deepstack_layers");
    let key_rms_eps = format!("{key_prefix}.attention.layer_norm_rms_epsilon");
    let key_softcap = format!("{key_prefix}.final_logit_softcapping");
    let key_rope_swa = format!("{key_prefix}.rope.freq_base_swa");
    let key_expert_count = format!("{key_prefix}.expert_count");
    let key_expert_used_count = format!("{key_prefix}.expert_used_count");
    let key_expert_ffn = format!("{key_prefix}.expert_feed_forward_length");
    let key_expert_shared_ffn = format!("{key_prefix}.expert_shared_feed_forward_length");
    let key_ssm_conv_kernel = format!("{key_prefix}.ssm.conv_kernel");
    let key_ssm_inner_size = format!("{key_prefix}.ssm.inner_size");
    let key_ssm_state_size = format!("{key_prefix}.ssm.state_size");
    let key_ssm_time_step_rank = format!("{key_prefix}.ssm.time_step_rank");
    let key_ssm_group_count = format!("{key_prefix}.ssm.group_count");

    let mut config = Config {
        dim: get_gguf_int_from_map(&gguf.kv, &key_dim, 4096) as usize,
        input_embedding_dim: 0,
        n_deepstack_layers: get_gguf_int_from_map(&gguf.kv, &key_n_deepstack, 0) as usize,
        hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_hidden, 11008) as usize,
        expert_hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_expert_ffn, 0) as usize,
        shared_expert_hidden_dim: get_gguf_int_from_map(&gguf.kv, &key_expert_shared_ffn, 0)
            as usize,
        n_layers: get_gguf_int_from_map(&gguf.kv, &key_layers, 32) as usize,
        n_heads: get_gguf_int_from_map(&gguf.kv, &key_heads, 32) as usize,
        n_kv_heads: 0,
        n_experts: get_gguf_int_from_map(&gguf.kv, &key_expert_count, 0) as usize,
        n_experts_used: get_gguf_int_from_map(&gguf.kv, &key_expert_used_count, 0) as usize,
        moe_n_group: 1,
        moe_topk_group: 1,
        moe_norm_topk_prob: false,
        moe_routed_scaling_factor: 1.0,
        vocab_size: get_gguf_int_from_map(&gguf.kv, &key_vocab, 32000) as usize,
        seq_len: get_gguf_int_from_map(&gguf.kv, &key_ctx, 2048) as usize,
        rope_theta: 0.0,
        head_dim: 0,
        rope_dim: 0,
        rope_sections: [0; 4],
        is_gemma3: identity.family == ModelFamily::Gemma,
        is_qwen2: identity.family == ModelFamily::Qwen2 || identity.family == ModelFamily::Qwen35,
        is_qwen35: identity.family == ModelFamily::Qwen35,
        is_qwen3vl: identity.family == ModelFamily::Qwen3Vl,
        is_qwen3moe: identity.family == ModelFamily::Qwen3Moe || is_qwen3vlmoe_arch,
        is_qwen3next: identity.family == ModelFamily::Qwen3Next
            || identity.family == ModelFamily::Qwen35,
        capabilities,
        final_logit_softcapping: get_gguf_float_from_map(&gguf.kv, &key_softcap, 0.0),
        rms_norm_eps: get_gguf_float_from_map(&gguf.kv, &key_rms_eps, 1e-6),
        rope_theta_swa: get_gguf_float_from_map(&gguf.kv, &key_rope_swa, 10_000.0),
        swa_pattern: 6,
        ssm_conv_kernel: get_gguf_int_from_map(&gguf.kv, &key_ssm_conv_kernel, 0) as usize,
        ssm_inner_size: get_gguf_int_from_map(&gguf.kv, &key_ssm_inner_size, 0) as usize,
        ssm_state_size: get_gguf_int_from_map(&gguf.kv, &key_ssm_state_size, 0) as usize,
        ssm_time_step_rank: get_gguf_int_from_map(&gguf.kv, &key_ssm_time_step_rank, 0) as usize,
        ssm_group_count: get_gguf_int_from_map(&gguf.kv, &key_ssm_group_count, 0) as usize,
    };

    let deepstack_multiplier = config
        .n_deepstack_layers
        .checked_add(1)
        .ok_or_else(|| "deepstack layer count overflow".to_string())?;
    config.input_embedding_dim = config
        .dim
        .checked_mul(deepstack_multiplier)
        .ok_or_else(|| "input embedding dimension overflow".to_string())?;

    if config.is_qwen3moe || (config.is_qwen3next && config.n_experts > 0) {
        qwen::finalize_moe_config(&mut config)?;
        if config.is_qwen3moe {
            qwen::apply_qwen3moe_defaults(&mut config);
        }
    }
    if config.is_qwen3next {
        qwen::validate_qwen3next(&mut config)?;
        if config.n_experts == 0 {
            config.moe_norm_topk_prob = false;
            config.moe_routed_scaling_factor = 1.0;
        }
    }

    config.n_kv_heads =
        get_gguf_int_from_map(&gguf.kv, &key_kv_heads, config.n_heads as i64) as usize;

    let default_rope_theta = if identity.family == ModelFamily::Gemma {
        gemma::default_rope_theta()
    } else {
        llama::default_rope_theta()
    };
    config.rope_theta = get_gguf_float_from_map(&gguf.kv, &key_rope, default_rope_theta);
    config.head_dim = get_gguf_int_from_map(
        &gguf.kv,
        &key_head_dim,
        (config.dim / config.n_heads) as i64,
    ) as usize;
    config.rope_dim =
        get_gguf_int_from_map(&gguf.kv, &key_rope_dim, config.head_dim as i64) as usize;
    if config.rope_dim == 0 || config.rope_dim > config.head_dim || (config.rope_dim & 1) != 0 {
        config.rope_dim = config.head_dim;
    }
    if let Some(raw_sections) = get_gguf_i64_array_from_map(&gguf.kv, &key_rope_sections) {
        let mut sections = [0usize; 4];
        for (idx, value) in raw_sections.iter().take(4).enumerate() {
            sections[idx] = if *value > 0 { *value as usize } else { 0 };
        }
        config.rope_sections = sections;
    }

    if !gguf.vocab_tokens.is_empty() && config.vocab_size != gguf.vocab_tokens.len() {
        if debug_mode {
            eprintln!(
                "Note: Updating vocab_size from {} to {} based on GGUF vocabulary",
                config.vocab_size,
                gguf.vocab_tokens.len()
            );
        }
        config.vocab_size = gguf.vocab_tokens.len();
    }

    if debug_mode {
        eprintln!(
            "Config: dim={}, input_embedding_dim={}, n_deepstack_layers={}, hidden_dim={}, expert_hidden_dim={}, n_layers={}, n_heads={}, n_kv_heads={}, vocab_size={}, seq_len={}",
            config.dim,
            config.input_embedding_dim,
            config.n_deepstack_layers,
            config.hidden_dim,
            config.expert_hidden_dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size,
            config.seq_len
        );
        eprintln!(
            "RoPE theta: {}, head_dim: {}, rope_dim: {}, rope_sections={:?}",
            config.rope_theta, config.head_dim, config.rope_dim, config.rope_sections
        );
        eprintln!(
            "Multimodal backend: {}, native image={}, native video={}, native audio={}",
            config.capabilities.multimodal_backend.as_str(),
            config.capabilities.supports_native_image,
            config.capabilities.supports_native_video,
            config.capabilities.supports_native_audio
        );
        if config.is_gemma3 {
            gemma::print_config_debug(&config);
        } else if config.is_qwen3moe {
            qwen::print_qwen3moe_debug(&config);
        } else if config.is_qwen3next {
            qwen::print_qwen3next_debug(&config);
        }
    }

    Ok(config)
}

pub(crate) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    config: &Config,
    prompt: &str,
    system_prompt: &str,
    image_count: usize,
    think_mode: ThinkMode,
) -> Vec<i32> {
    if config.is_gemma3 {
        gemma::encode_chat_prompt(tokenizer, prompt, system_prompt)
    } else if config.is_qwen3moe || config.is_qwen3next || config.is_qwen3vl || config.is_qwen35 {
        qwen::encode_qwen3_chat(tokenizer, prompt, system_prompt, image_count, think_mode)
    } else if config.is_qwen2 {
        qwen::encode_qwen2_chat(tokenizer, prompt, system_prompt)
    } else {
        llama::encode_chat_prompt(tokenizer, prompt, system_prompt)
    }
}

fn join_request_text(parts: &[ContentPart]) -> String {
    let mut text_parts: Vec<&str> = Vec::new();
    for part in parts {
        if let ContentPart::Text(text) = part {
            text_parts.push(text);
        }
    }
    text_parts.join("\n")
}

pub(crate) fn encode_generation_request(
    tokenizer: &mut Tokenizer,
    config: &Config,
    request: &GenerationRequest,
    think_mode: ThinkMode,
) -> EncodedPrompt {
    if config.is_qwen3moe || config.is_qwen3next || config.is_qwen3vl || config.is_qwen35 {
        return qwen::encode_qwen3_request(tokenizer, request, think_mode);
    }

    let prompt = join_request_text(&request.parts);
    let token_ids = if config.is_gemma3 {
        gemma::encode_chat_prompt(tokenizer, &prompt, &request.system_prompt)
    } else if config.is_qwen2 {
        qwen::encode_qwen2_chat(tokenizer, &prompt, &request.system_prompt)
    } else {
        llama::encode_chat_prompt(tokenizer, &prompt, &request.system_prompt)
    };
    EncodedPrompt::from_token_ids(token_ids)
}
