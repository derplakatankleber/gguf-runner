use crate::{
    get_gguf_float_from_map, get_gguf_int_from_map, get_gguf_string_from_map, Config, GGUFFile,
};

struct ModelIdentity {
    key_prefix: String,
    is_gemma3: bool,
    is_qwen2: bool,
    is_qwen3moe: bool,
    is_qwen3next: bool,
}

fn detect_model_identity(gguf: &GGUFFile, debug_mode: bool) -> Result<ModelIdentity, String> {
    let arch = get_gguf_string_from_map(&gguf.kv, "general.architecture").unwrap_or("llama");
    if debug_mode {
        eprintln!("Model architecture: {arch}");
    }

    let mut identity = ModelIdentity {
        key_prefix: "llama".to_string(),
        is_gemma3: false,
        is_qwen2: false,
        is_qwen3moe: false,
        is_qwen3next: false,
    };

    if arch == "gemma3" || arch == "gemma2" || arch == "gemma" {
        identity.is_gemma3 = true;
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
    } else if arch == "qwen3moe" || arch.starts_with("qwen3moe") {
        identity.is_qwen3moe = true;
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
        identity.is_qwen3next = true;
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
        identity.is_qwen2 = true;
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

    Ok(identity)
}

pub(crate) fn build_config_from_gguf(gguf: &GGUFFile, debug_mode: bool) -> Result<Config, String> {
    let identity = detect_model_identity(gguf, debug_mode)?;
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
    let key_head_dim = format!("{key_prefix}.attention.key_length");
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
        is_gemma3: identity.is_gemma3,
        is_qwen2: identity.is_qwen2,
        is_qwen3moe: identity.is_qwen3moe,
        is_qwen3next: identity.is_qwen3next,
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

    if config.is_qwen3moe || config.is_qwen3next {
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
        if config.is_qwen3moe {
            // Qwen3 MoE routing defaults from official config.json.
            config.moe_n_group = 8;
            config.moe_topk_group = 4;
            config.moe_norm_topk_prob = true;
            config.moe_routed_scaling_factor = 2.5;
        }
        if config.is_qwen3next {
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
            if config.ssm_inner_size % config.ssm_time_step_rank != 0 {
                return Err(format!(
                    "qwen3next invalid SSM metadata: inner_size {} not divisible by time_step_rank {}",
                    config.ssm_inner_size, config.ssm_time_step_rank
                ));
            }
            if config.ssm_time_step_rank % config.ssm_group_count != 0 {
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
        }
    }

    config.n_kv_heads =
        get_gguf_int_from_map(&gguf.kv, &key_kv_heads, config.n_heads as i64) as usize;

    let default_rope_theta = if config.is_gemma3 {
        1_000_000.0
    } else {
        500_000.0
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
            "Config: dim={}, hidden_dim={}, expert_hidden_dim={}, n_layers={}, n_heads={}, n_kv_heads={}, vocab_size={}, seq_len={}",
            config.dim,
            config.hidden_dim,
            config.expert_hidden_dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size,
            config.seq_len
        );
        eprintln!(
            "RoPE theta: {}, head_dim: {}, rope_dim: {}",
            config.rope_theta, config.head_dim, config.rope_dim
        );
        if config.is_gemma3 {
            eprintln!(
                "Gemma3: rms_norm_eps={}, final_logit_softcapping={}",
                config.rms_norm_eps, config.final_logit_softcapping
            );
        } else if config.is_qwen3moe {
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
        } else if config.is_qwen3next {
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
    }

    Ok(config)
}

pub(crate) fn apply_context_size_overrides(
    config: &mut Config,
    context_size: usize,
    debug_mode: bool,
) {
    if context_size > 0 {
        config.seq_len = context_size;
    } else if (config.is_qwen3moe || config.is_qwen3next) && config.seq_len > 8192 {
        if debug_mode {
            eprintln!(
                "Clamping context length from {} to 8192 for qwen3 model; pass -context_size to override",
                config.seq_len
            );
        }
        config.seq_len = 8192;
    }
}
