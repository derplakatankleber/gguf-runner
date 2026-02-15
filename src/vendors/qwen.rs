use crate::engine::types::{Config, Tokenizer};

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
    config.moe_routed_scaling_factor = 2.5;
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
) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = system_prompt.trim();

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

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

    let rendered = if sys.is_empty() {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        format!(
            "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    };
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}
