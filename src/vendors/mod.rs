mod gemma;
mod llama;
mod qwen2;
mod qwen3;
mod qwen35;
mod qwen3next;
mod qwen3vl;
mod qwen_common;

use crate::engine::io::{
    get_gguf_float_from_map, get_gguf_i64_array_from_map, get_gguf_int_from_map,
    get_gguf_string_from_map,
};
use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GGUFFile, GenerationRequest, ModelCapabilities,
    MultimodalBackend, ThinkMode, Tokenizer, VendorTokenizerPolicy,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChatRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChatMessage {
    pub(crate) role: ChatRole,
    pub(crate) content: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorDecodePolicy {
    pub(crate) parse_think_tags: bool,
    pub(crate) stop_token_literals: &'static [&'static str],
    pub(crate) deterministic_loop_guard: bool,
    pub(crate) deterministic_loop_guard_min_generated_tokens: usize,
    pub(crate) recover_early_endoftext_once: bool,
    pub(crate) early_endoftext_recover_max_tokens: usize,
    pub(crate) hidden_think_token_cap_base: usize,
    pub(crate) retry_without_think_when_no_post_think_text: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorRuntimeDebugPolicy {
    pub(crate) native_context_label: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MmprojFilenameScoreHint {
    pub(crate) token: &'static str,
    pub(crate) backend: MultimodalBackend,
    pub(crate) match_score: i32,
    pub(crate) mismatch_score: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorDetailCropPolicy {
    pub(crate) enabled: bool,
    pub(crate) max_layers: usize,
    pub(crate) note_text: &'static str,
    pub(crate) temp_file_prefix: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorMultimodalPolicy {
    pub(crate) image_prompt_suffix: &'static str,
    pub(crate) detail_crop: VendorDetailCropPolicy,
    pub(crate) mmproj_filename_score_hints: &'static [MmprojFilenameScoreHint],
    pub(crate) missing_sidecar_hint: &'static str,
}

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

fn normalize_alpha_num(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
}

fn qwen_mmproj_family_markers(mmproj: &GGUFFile) -> (bool, bool) {
    let mut combined = String::new();
    for key in [
        "general.name",
        "general.basename",
        "general.finetune",
        "general.base_model.0.name",
        "general.base_model.0.repo_url",
        "general.repo_url",
    ] {
        if let Some(value) = get_gguf_string_from_map(&mmproj.kv, key) {
            combined.push_str(value);
            combined.push(' ');
        }
    }
    let normalized = normalize_alpha_num(&combined);
    let has_qwen3vl = normalized.contains("qwen3vl");
    let has_qwen35 = normalized.contains("qwen35") || normalized.contains("qwen3p5");
    (has_qwen3vl, has_qwen35)
}

pub(crate) fn validate_mmproj_for_backend(cfg: &Config, mmproj: &GGUFFile) -> Result<(), String> {
    let arch = get_gguf_string_from_map(&mmproj.kv, "general.architecture").unwrap_or_default();
    let projector = get_gguf_string_from_map(&mmproj.kv, "clip.projector_type").unwrap_or_default();
    let projection_dim =
        get_gguf_int_from_map(&mmproj.kv, "clip.vision.projection_dim", 0) as usize;
    if projection_dim == 0 {
        return Err(
            "mmproj is missing clip.vision.projection_dim; cannot verify text/vision dim compatibility"
                .to_string(),
        );
    }
    if projection_dim != cfg.dim {
        return Err(format!(
            "mmproj/text dim mismatch: clip.vision.projection_dim={} but model embedding dim={}. hint: this mmproj likely belongs to a different checkpoint",
            projection_dim, cfg.dim
        ));
    }

    match cfg.capabilities.multimodal_backend {
        MultimodalBackend::Gemma3 => {
            if arch != "clip" || projector != "gemma3" {
                return Err(
                    "unsupported mmproj for this runner: expected clip.projector_type='gemma3'"
                        .to_string(),
                );
            }
            Ok(())
        }
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
            if arch != "clip" || projector != "qwen3vl_merger" {
                return Err(
                    "unsupported mmproj for this runner: expected clip.projector_type='qwen3vl_merger'"
                        .to_string(),
                );
            }
            let (has_qwen3vl, has_qwen35) = qwen_mmproj_family_markers(mmproj);
            match cfg.capabilities.multimodal_backend {
                MultimodalBackend::Qwen35 => {
                    if has_qwen3vl && !has_qwen35 {
                        return Err("incompatible mmproj family for qwen35 backend: sidecar metadata indicates Qwen3-VL; use a Qwen3.5 mmproj from the same checkpoint family".to_string());
                    }
                    if mmproj
                        .tensors
                        .iter()
                        .any(|tensor| tensor.name.starts_with("v.deepstack."))
                    {
                        return Err("incompatible mmproj family for qwen35 backend: deepstack vision tensors detected (Qwen3-VL style); use a Qwen3.5 mmproj sidecar".to_string());
                    }
                }
                MultimodalBackend::Qwen3Vl => {
                    if has_qwen35 && !has_qwen3vl {
                        return Err("incompatible mmproj family for qwen3vl backend: sidecar metadata indicates Qwen3.5; use a Qwen3-VL mmproj from the same checkpoint family".to_string());
                    }
                }
                _ => {}
            }
            Ok(())
        }
        _ => Err(format!(
            "external vision mmproj is unsupported for backend '{}'",
            cfg.capabilities.multimodal_backend.as_str()
        )),
    }
}

fn detect_model_capabilities(
    gguf: &GGUFFile,
    family: ModelFamily,
    debug_mode: bool,
) -> ModelCapabilities {
    let multimodal_backend = match family {
        ModelFamily::Gemma
            if get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or_default()
                == "gemma3" =>
        {
            MultimodalBackend::Gemma3
        }
        ModelFamily::Qwen3Vl => MultimodalBackend::Qwen3Vl,
        ModelFamily::Qwen35 => MultimodalBackend::Qwen35,
        _ => MultimodalBackend::None,
    };

    if multimodal_backend == MultimodalBackend::None {
        return ModelCapabilities::default();
    }

    let (has_image_tokens, has_video_tokens, has_audio_tokens) = match multimodal_backend {
        MultimodalBackend::Gemma3 => (
            has_vocab_token(gguf, "<start_of_image>") && has_vocab_token(gguf, "<end_of_image>"),
            false,
            false,
        ),
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => (
            has_vocab_token(gguf, "<|vision_start|>")
                && has_vocab_token(gguf, "<|vision_end|>")
                && has_vocab_token(gguf, "<|image_pad|>"),
            has_vocab_token(gguf, "<|vision_start|>")
                && has_vocab_token(gguf, "<|vision_end|>")
                && has_vocab_token(gguf, "<|video_pad|>"),
            has_vocab_token(gguf, "<|audio_pad|>"),
        ),
        MultimodalBackend::None => (false, false, false),
    };
    let has_vision_tensors = has_tensor_with_any_prefix(gguf, VISION_TENSOR_PREFIXES);
    let has_audio_tensors = has_tensor_with_any_prefix(gguf, AUDIO_TENSOR_PREFIXES);

    let capabilities = ModelCapabilities {
        multimodal_backend,
        supports_native_image: has_image_tokens && has_vision_tensors,
        supports_native_video: has_video_tokens && has_vision_tensors,
        supports_native_audio: has_audio_tokens && has_audio_tensors,
    };

    if debug_mode {
        eprintln!(
            "Multimodal capability probe (backend={}): tokens(image={} video={} audio={}) tensors(vision={} audio={}) native(image={} video={} audio={})",
            capabilities.multimodal_backend.as_str(),
            has_image_tokens,
            has_video_tokens,
            has_audio_tokens,
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
    let key_full_attention_interval = format!("{key_prefix}.full_attention_interval");

    let chat_template = get_gguf_string_from_map(&gguf.kv, "tokenizer.chat_template")
        .unwrap_or_default()
        .to_ascii_lowercase();

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
        online_attn_fusion: false,
        qwen_chat_template_contains_think: chat_template.contains("<think>"),
        qwen_chat_template_has_builtin_system: chat_template.contains("you are qwen"),
        capabilities,
        final_logit_softcapping: get_gguf_float_from_map(&gguf.kv, &key_softcap, 0.0),
        rms_norm_eps: get_gguf_float_from_map(&gguf.kv, &key_rms_eps, 1e-6),
        rope_theta_swa: get_gguf_float_from_map(&gguf.kv, &key_rope_swa, 10_000.0),
        swa_pattern: get_gguf_int_from_map(&gguf.kv, &key_full_attention_interval, 6) as usize,
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
        qwen3::finalize_moe_config(&mut config)?;
        if config.is_qwen3moe {
            qwen3::apply_qwen3moe_defaults(&mut config);
        }
    }
    if config.is_qwen3next {
        qwen3next::validate_qwen3next(&mut config)?;
        if config.n_experts == 0 {
            config.moe_norm_topk_prob = false;
            config.moe_routed_scaling_factor = 1.0;
        }
    }

    config.online_attn_fusion = config.is_qwen35;

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
            qwen3::print_qwen3moe_debug(&config);
        } else if config.is_qwen3next {
            qwen3next::print_qwen3next_debug(&config);
            eprintln!(
                "Qwen template hints: contains_think={}, has_builtin_system={}",
                config.qwen_chat_template_contains_think,
                config.qwen_chat_template_has_builtin_system
            );
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
    } else if config.is_qwen35 {
        qwen35::encode_chat_prompt(tokenizer, prompt, system_prompt, image_count, think_mode)
    } else if config.is_qwen3vl {
        qwen3vl::encode_chat_prompt(tokenizer, prompt, system_prompt, image_count, think_mode)
    } else if config.is_qwen3next {
        qwen3next::encode_chat_prompt(
            tokenizer,
            config,
            prompt,
            system_prompt,
            image_count,
            think_mode,
        )
    } else if config.is_qwen3moe {
        qwen3::encode_chat_prompt(tokenizer, prompt, system_prompt, image_count, think_mode)
    } else if config.is_qwen2 {
        qwen2::encode_chat_prompt(tokenizer, prompt, system_prompt)
    } else {
        llama::encode_chat_prompt(tokenizer, prompt, system_prompt)
    }
}

pub(crate) fn encode_chat_messages(
    tokenizer: &mut Tokenizer,
    cfg: &Config,
    messages: &[ChatMessage],
    system_prompt: &str,
    think_mode: ThinkMode,
) -> Vec<i32> {
    if cfg.is_gemma3 {
        gemma::encode_chat_messages(tokenizer, messages, system_prompt)
    } else if cfg.is_qwen35 {
        qwen35::encode_chat_messages(tokenizer, messages, system_prompt, think_mode)
    } else if cfg.is_qwen3vl {
        qwen3vl::encode_chat_messages(tokenizer, messages, system_prompt, think_mode)
    } else if cfg.is_qwen3next {
        qwen3next::encode_chat_messages(tokenizer, cfg, messages, system_prompt, think_mode)
    } else if cfg.is_qwen3moe {
        qwen3::encode_chat_messages(tokenizer, messages, system_prompt, think_mode)
    } else if cfg.is_qwen2 {
        qwen2::encode_chat_messages(tokenizer, messages, system_prompt)
    } else {
        llama::encode_chat_messages(tokenizer, messages, system_prompt)
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
    if config.is_gemma3 {
        return gemma::encode_generation_request(tokenizer, request);
    }
    if config.is_qwen35 {
        return qwen35::encode_generation_request(tokenizer, request, think_mode);
    }
    if config.is_qwen3vl {
        return qwen3vl::encode_generation_request(tokenizer, request, think_mode);
    }
    if config.is_qwen3next {
        return qwen3next::encode_generation_request(tokenizer, config, request, think_mode);
    }
    if config.is_qwen3moe {
        return qwen3::encode_generation_request(tokenizer, request, think_mode);
    }

    let prompt = join_request_text(&request.parts);
    let token_ids = if config.is_qwen2 {
        qwen2::encode_chat_prompt(tokenizer, &prompt, &request.system_prompt)
    } else {
        llama::encode_chat_prompt(tokenizer, &prompt, &request.system_prompt)
    };
    EncodedPrompt::from_token_ids(token_ids)
}

pub(crate) fn decode_policy(config: &Config) -> VendorDecodePolicy {
    if config.is_qwen35 {
        qwen35::decode_policy()
    } else if config.is_qwen3vl {
        qwen3vl::decode_policy()
    } else if config.is_qwen3next {
        qwen3next::decode_policy(config)
    } else if config.is_qwen3moe {
        qwen3::decode_policy()
    } else if config.is_qwen2 {
        qwen2::decode_policy()
    } else if config.is_gemma3 {
        gemma::decode_policy()
    } else {
        llama::decode_policy()
    }
}

pub(crate) fn tokenizer_policy(config: &Config) -> VendorTokenizerPolicy {
    if config.is_qwen35 {
        qwen35::tokenizer_policy()
    } else if config.is_qwen3vl {
        qwen3vl::tokenizer_policy()
    } else if config.is_qwen3next {
        qwen3next::tokenizer_policy()
    } else if config.is_qwen3moe {
        qwen3::tokenizer_policy()
    } else if config.is_qwen2 {
        qwen2::tokenizer_policy()
    } else if config.is_gemma3 {
        gemma::tokenizer_policy()
    } else {
        llama::tokenizer_policy()
    }
}

pub(crate) fn multimodal_policy(config: &Config) -> VendorMultimodalPolicy {
    if config.is_qwen35 {
        qwen35::multimodal_policy()
    } else if config.is_qwen3vl {
        qwen3vl::multimodal_policy()
    } else if config.is_qwen3next {
        qwen3next::multimodal_policy()
    } else if config.is_qwen3moe {
        qwen3::multimodal_policy()
    } else if config.is_qwen2 {
        qwen2::multimodal_policy()
    } else if config.is_gemma3 {
        gemma::multimodal_policy()
    } else {
        llama::multimodal_policy()
    }
}

pub(crate) fn runtime_debug_policy(config: &Config) -> VendorRuntimeDebugPolicy {
    if config.is_qwen35 {
        qwen35::runtime_debug_policy()
    } else if config.is_qwen3vl {
        qwen3vl::runtime_debug_policy()
    } else if config.is_qwen3next {
        qwen3next::runtime_debug_policy()
    } else if config.is_qwen3moe {
        qwen_common::runtime_debug_policy()
    } else if config.is_gemma3 {
        gemma::runtime_debug_policy()
    } else {
        llama::runtime_debug_policy()
    }
}
