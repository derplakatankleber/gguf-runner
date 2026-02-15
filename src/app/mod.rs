use crate::cli::CliOptions;
use crate::engine::io::parse_gguf_file;
use crate::engine::kernels::{argmax, sample, softmax, TopKSampler};
use crate::engine::profiling::{
    print_profile_report, prof_end, prof_start, profiling_reset, record_forward_pass,
    set_profiling_enabled, PROF_TRANSFORMER_NS,
};
use crate::engine::switches::{
    init_runtime_config, par_attn_min_heads, par_matmul_chunk_rows, par_matmul_min_rows,
    par_qwen3next_min_heads, RuntimeSwitchConfig,
};
use crate::engine::types::{GEMMA3_END_TURN, XorShiftRng};
use std::cmp::Ordering;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn time_in_ms() -> i64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() * 1000 + (now.subsec_nanos() as u64 / 1_000_000)) as i64
}

pub(crate) fn run() -> Result<(), String> {
    let cli = CliOptions::parse()?;
    let runtime_switch_config = RuntimeSwitchConfig {
        par_matmul_min_rows: cli.par_matmul_min_rows,
        par_matmul_chunk_rows: cli.par_matmul_chunk_rows,
        par_attn_min_heads: cli.par_attn_min_heads,
        par_qwen3next_min_heads: cli.par_qwen3next_min_heads,
        #[cfg(target_arch = "aarch64")]
        aarch64_dotprod_q8: cli.aarch64_dotprod_q8,
        #[cfg(target_arch = "aarch64")]
        aarch64_qk_mr4: cli.aarch64_qk_mr4,
        #[cfg(target_arch = "x86_64")]
        x86_avx2: cli.x86_avx2,
        #[cfg(target_arch = "x86_64")]
        x86_f16c: cli.x86_f16c,
        #[cfg(target_arch = "x86_64")]
        x86_qk_mr4: cli.x86_qk_mr4,
        layer_debug: cli.layer_debug,
        layer_debug_pos: cli.layer_debug_pos,
    };
    init_runtime_config(&runtime_switch_config);
    let run_started = Instant::now();

    let temperature = cli.temperature;
    let top_k = cli.top_k;
    let top_p = cli.top_p;
    let mut max_tokens = cli.max_tokens;
    let context_size = cli.context_size;
    let rayon_threads = cli.threads;
    let system_prompt = cli.system_prompt;
    let checkpoint = cli.model;
    let model_url = cli.url;
    let prompt = cli.prompt;
    let profiling_mode = cli.profiling;
    let show_tokens = cli.show_tokens;
    let show_timings = cli.show_timings;
    let debug_mode = cli.debug;

    if debug_mode {
        eprintln!("Loading GGUF model: {checkpoint}");
        eprintln!("Sampling: temperature={temperature}, top_k={top_k}, top_p={top_p}");
    }

    let gguf = parse_gguf_file(&checkpoint, model_url.as_deref(), debug_mode)?;
    let lazy_debug_loader = gguf.lazy_loader.as_ref().map(Arc::clone);
    let mut next_lazy_debug_ms = time_in_ms() + 2_000;

    if debug_mode {
        eprintln!(
            "GGUF metadata: version={}, tensors={}, kv={}, tensor_data_start={} bytes",
            gguf.version, gguf.n_tensors, gguf.n_kv, gguf.tensor_data_start
        );
        if let Some(loader) = &lazy_debug_loader {
            eprintln!("{}", loader.debug_stats_line());
        }
    }

    let mut config = crate::vendors::build_config_from_gguf(&gguf, debug_mode)?;

    let mut tokenizer = crate::engine::tokenizer::init_tokenizer_from_gguf(
        &gguf,
        &mut config,
        debug_mode,
    )?;
    tokenizer.use_sentencepiece = config.is_gemma3;

    crate::engine::runtime::apply_context_size_overrides(&mut config, context_size, debug_mode);
    if max_tokens == 0 || max_tokens > config.seq_len {
        max_tokens = config.seq_len;
    }

    if let Some(n_threads) = rayon_threads {
        crate::engine::runtime::configure_rayon_threads(n_threads, debug_mode);
    }

    set_profiling_enabled(profiling_mode);
    if profiling_mode {
        profiling_reset();
    }

    if debug_mode {
        eprintln!(
            "Parallel thresholds: matmul_min_rows={}, matmul_chunk_rows={}, attn_min_heads={}, qwen3next_min_heads={}",
            par_matmul_min_rows(),
            par_matmul_chunk_rows(),
            par_attn_min_heads(),
            par_qwen3next_min_heads()
        );
    }

    let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, debug_mode)?;
    let mut state = crate::engine::runtime::malloc_run_state(&config);

    let use_chat_template = true;
    let mut prompt_tokens: Vec<i32> = if use_chat_template {
        crate::vendors::encode_chat_prompt(&mut tokenizer, &config, &prompt, &system_prompt)
    } else {
        let mut t = Vec::new();
        tokenizer.bpe_encode(&prompt, &mut t);
        t
    };

    if prompt_tokens.is_empty() {
        prompt_tokens.push(tokenizer.bos_token);
    }
    if prompt_tokens.len() > config.seq_len {
        prompt_tokens.truncate(config.seq_len);
    }
    if debug_mode {
        eprintln!("Prompt tokens: {}", prompt_tokens.len());
        let preview = prompt_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("Prompt token ids: [{preview}]");
    }

    let mut token = prompt_tokens[0];
    let mut next: i32;
    let mut pos = 0usize;

    let mut rng = XorShiftRng::new(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    );
    let mut topk_sampler = TopKSampler::new();
    let mut warned_top_p_without_top_k = false;

    let mut recent_tokens = [0i32; 64];
    let mut recent_count = 0usize;
    let repetition_penalty = 1.0f32;
    let mut pending_newline = false;

    let gemma3_end_turn = if config.is_gemma3 {
        tokenizer
            .find_special_token("<end_of_turn>")
            .unwrap_or(GEMMA3_END_TURN)
    } else {
        -1
    };
    let qwen_im_end = if config.is_qwen2 || config.is_qwen3moe || config.is_qwen3next {
        tokenizer.find_special_token("<|im_end|>").unwrap_or(-1)
    } else {
        -1
    };

    let mut start = 0i64;

    while pos < max_tokens {
        if token < 0 || token as usize >= config.vocab_size {
            return Err(format!("token id out of bounds: {token}"));
        }

        let prof_t0 = prof_start();
        crate::engine::runtime::transformer(
            token as usize,
            pos,
            &config,
            &mut state,
            &weights,
            gguf.mapped.as_slice(),
        )?;
        prof_end(&PROF_TRANSFORMER_NS, prof_t0);
        if profiling_mode {
            record_forward_pass();
        }

        if debug_mode {
            if let Some(loader) = &lazy_debug_loader {
                let now = time_in_ms();
                if now >= next_lazy_debug_ms {
                    eprintln!("{}", loader.debug_stats_line());
                    next_lazy_debug_ms = now + 2_000;
                }
            }
        }

        if debug_mode
            && pos >= prompt_tokens.len().saturating_sub(1)
            && pos < prompt_tokens.len() + 3
        {
            let mut top: Vec<(usize, f32)> = state.logits[..config.vocab_size]
                .iter()
                .copied()
                .enumerate()
                .collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            eprint!("[DEBUG pos={pos}] Top 5 logits: ");
            for (id, v) in top.into_iter().take(5) {
                let decoded = tokenizer
                    .decode_token(id as i32)
                    .unwrap_or_else(|| "?".to_string())
                    .replace('\n', "\\n")
                    .replace('\r', "\\r");
                eprint!("{id}({v:.2},\"{decoded}\") ");
            }
            eprintln!();
        }

        if pos < prompt_tokens.len().saturating_sub(1) {
            next = prompt_tokens[pos + 1];
        } else {
            for i in 0..recent_count {
                let tok = recent_tokens[i];
                if tok >= 0 && (tok as usize) < config.vocab_size {
                    let idx = tok as usize;
                    if state.logits[idx] > 0.0 {
                        state.logits[idx] /= repetition_penalty;
                    } else {
                        state.logits[idx] *= repetition_penalty;
                    }
                }
            }

            if temperature == 0.0 {
                next = argmax(&state.logits[..config.vocab_size]) as i32;
            } else if top_k > 0 {
                next = topk_sampler.sample_top_k_top_p(
                    &state.logits[..config.vocab_size],
                    temperature,
                    top_k,
                    top_p,
                    &mut rng,
                ) as i32;
            } else {
                if top_p < 1.0 && debug_mode && !warned_top_p_without_top_k {
                    eprintln!("Note: -top_p is ignored unless -top_k > 0");
                    warned_top_p_without_top_k = true;
                }
                for q in 0..config.vocab_size {
                    state.logits[q] /= temperature;
                }
                softmax(&mut state.logits[..config.vocab_size], config.vocab_size);
                next = sample(&state.logits[..config.vocab_size], &mut rng) as i32;
            }

            if recent_count < 64 {
                recent_tokens[recent_count] = next;
                recent_count += 1;
            } else {
                for i in 0..63 {
                    recent_tokens[i] = recent_tokens[i + 1];
                }
                recent_tokens[63] = next;
            }
        }

        if pos >= prompt_tokens.len().saturating_sub(1)
            && next != tokenizer.eot_token
            && next != tokenizer.eos_token
        {
            if let Some(decoded) = tokenizer.decode_token(next) {
                if decoded == "\n" {
                    pending_newline = true;
                } else {
                    if pending_newline {
                        print!("\n");
                        pending_newline = false;
                    }
                    print!("{decoded}");
                    let _ = io::stdout().flush();
                }
            }
        }

        token = next;
        pos += 1;

        if start == 0 {
            start = time_in_ms();
        }

        if pos >= prompt_tokens.len().saturating_sub(1) {
            if token == tokenizer.eos_token || token == tokenizer.eot_token {
                break;
            }
            if config.is_gemma3 && token == gemma3_end_turn {
                break;
            }
            if (config.is_qwen2 || config.is_qwen3moe || config.is_qwen3next)
                && qwen_im_end >= 0
                && token == qwen_im_end
            {
                break;
            }
        }
    }

    let end = time_in_ms();
    if (debug_mode || show_tokens) && pos > 1 {
        let elapsed_ms = (end - start).max(1) as f64;
        eprintln!(
            "\nachieved tok/s: {:.3}",
            (pos - 1) as f64 / elapsed_ms * 1000.0
        );
    } else {
        println!();
    }

    if profiling_mode {
        print_profile_report();
    }

    if show_timings {
        eprintln!("overall runtime: {:.3}s", run_started.elapsed().as_secs_f64());
    }

    Ok(())
}
