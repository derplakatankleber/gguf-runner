use crate::engine::kernels::{
    accum, axpy_inplace, dot_f32_simd, finite_or_zero, l2_norm, matmul_f32_embeddings,
    matmul_quantized, matmul_quantized_rows, qwen3next_linear_attention_autoregressive, rmsnorm,
    rmsnorm_gemma, rmsnorm_inplace, rmsnorm_per_head_gemma_inplace, scale_slice_inplace,
    select_topk_softmax, sigmoidf, softmax,
};
use crate::engine::profiling::{prof_end, prof_start, PROF_ATTN_NS, PROF_FFN_NS, PROF_MOE_NS};
use crate::engine::switches::{layer_debug_enabled, layer_debug_pos, par_attn_min_heads};
use crate::engine::types::{Config, RunState, TransformerWeights};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

pub(crate) fn malloc_run_state(p: &Config) -> RunState {
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        p.dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let ssm_inner = p.ssm_inner_size;
    let ssm_k_heads = p.ssm_group_count;
    let ssm_v_heads = p.ssm_time_step_rank;
    let ssm_head_dim = p.ssm_state_size;
    let ssm_conv_dim = if p.is_qwen3next {
        ssm_inner + 2 * ssm_k_heads * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_hist = if p.is_qwen3next {
        p.ssm_conv_kernel.saturating_sub(1)
    } else {
        0
    };
    let ssm_state_stride = if p.is_qwen3next {
        ssm_v_heads * ssm_head_dim * ssm_head_dim
    } else {
        0
    };
    let ssm_conv_stride = if p.is_qwen3next {
        ssm_conv_hist * ssm_conv_dim
    } else {
        0
    };
    let max_dim = p.dim.max(q_dim);
    let ffn_dim = p
        .hidden_dim
        .max(p.expert_hidden_dim)
        .max(p.shared_expert_hidden_dim);
    let scratch_dim = ffn_dim.max(ssm_conv_dim).max(ssm_inner).max(ssm_head_dim);

    let rope_dim = if p.rope_dim > 0 {
        p.rope_dim
    } else {
        head_size
    };
    let rope_size = rope_dim / 2;
    let mut rope_freqs = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs.iter_mut().enumerate() {
        *freq = 1.0 / p.rope_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    let swa_theta = if p.rope_theta_swa > 0.0 {
        p.rope_theta_swa
    } else {
        10_000.0
    };
    let mut rope_freqs_swa = vec![0.0f32; rope_size];
    for (i, freq) in rope_freqs_swa.iter_mut().enumerate() {
        *freq = 1.0 / swa_theta.powf((i * 2) as f32 / rope_dim as f32);
    }

    RunState {
        x: vec![0.0; p.dim],
        xb: vec![0.0; max_dim],
        xb2: vec![0.0; p.dim],
        hb: vec![0.0; scratch_dim],
        hb2: vec![0.0; scratch_dim],
        moe_tmp: vec![0.0; p.dim],
        moe_logits: vec![0.0; p.n_experts],
        moe_topk_indices: vec![0usize; p.n_experts_used.max(1)],
        moe_topk_weights: vec![0.0f32; p.n_experts_used.max(1)],
        moe_scores: vec![0.0; p.n_experts],
        moe_selected_group: vec![true; p.moe_n_group.max(1)],
        moe_group_scores: vec![0.0; p.moe_n_group.max(1)],
        moe_group_rank: vec![0usize; p.moe_n_group.max(1)],
        q: vec![0.0; q_dim],
        k: vec![0.0; kv_dim],
        v: vec![0.0; kv_dim],
        ssm_qkv: vec![0.0; ssm_conv_dim],
        ssm_conv: vec![0.0; ssm_conv_dim],
        ssm_q: vec![0.0; ssm_inner],
        ssm_k: vec![0.0; ssm_inner],
        ssm_v: vec![0.0; ssm_inner],
        ssm_z: vec![0.0; ssm_inner],
        ssm_ba: vec![0.0; 2 * ssm_v_heads],
        ssm_gate_exp: vec![0.0; ssm_v_heads],
        ssm_beta: vec![0.0; ssm_v_heads],
        ssm_proj: vec![0.0; ssm_inner],
        ssm_kv_mem: vec![0.0; ssm_inner],
        ssm_delta: vec![0.0; ssm_inner],
        ssm_conv_state: vec![0.0; p.n_layers * ssm_conv_stride],
        ssm_state: vec![0.0; p.n_layers * ssm_state_stride],
        att: vec![0.0; p.n_heads * p.seq_len],
        logits: vec![0.0; p.vocab_size],
        key_cache: vec![0.0; p.n_layers * p.seq_len * kv_dim],
        value_cache: vec![0.0; p.n_layers * p.seq_len * kv_dim],
        rope_freqs,
        rope_freqs_swa,
        rope_cos: vec![0.0; rope_size],
        rope_sin: vec![0.0; rope_size],
        rope_cache_pos: -1,
        rope_cache_is_swa: -1,
        head_size,
        kv_dim,
        q_dim,
        kv_mul: p.n_heads / p.n_kv_heads,
        kv_cache_layer_size: p.seq_len * kv_dim,
        attn_scale: 1.0 / (head_size as f32).sqrt(),
        embed_scale: (p.dim as f32).sqrt(),
    }
}

pub(crate) fn transformer(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    let dim = p.dim;
    let hidden_dim = p.hidden_dim;
    let head_size = s.head_size;
    let kv_dim = s.kv_dim;
    let q_dim = s.q_dim;
    let kv_mul = s.kv_mul;
    let eps = if p.rms_norm_eps > 0.0 {
        p.rms_norm_eps
    } else {
        1e-5
    };
    let do_layer_debug =
        layer_debug_enabled() && layer_debug_pos().map_or(pos == 0, |p0| pos == p0);

    let emb_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
    s.x[..dim].copy_from_slice(emb_row);

    if p.is_gemma3 {
        scale_slice_inplace(&mut s.x[..dim], s.embed_scale);
    }

    for l in 0..p.n_layers {
        if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        let is_qwen3next_ssm_layer = p.is_qwen3next && w.attn_qkv[l].rows > 0;
        if is_qwen3next_ssm_layer {
            qwen3next_linear_attention_autoregressive(l, p, s, w, mapped, eps)?;
        } else {
            let attn_prof = prof_start();
            let mut qwen3next_packed_q_gate = false;
            if p.is_qwen3next {
                if w.wq[l].rows >= 2 * q_dim {
                    // Qwen3Next full-attn packs Q and gate interleaved per head:
                    // [q_head0, gate_head0, q_head1, gate_head1, ...]
                    matmul_quantized_rows(
                        &mut s.hb[..2 * q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        2 * q_dim,
                        mapped,
                    )?;
                    if p.n_heads >= par_attn_min_heads() {
                        let hb_src = &s.hb[..2 * q_dim];
                        s.q[..q_dim].par_chunks_mut(head_size).enumerate().for_each(
                            |(h, q_dst)| {
                                let src_base = h * 2 * head_size;
                                q_dst.copy_from_slice(&hb_src[src_base..src_base + head_size]);
                            },
                        );
                    } else {
                        for h in 0..p.n_heads {
                            let src_base = h * 2 * head_size;
                            let dst_base = h * head_size;
                            s.q[dst_base..dst_base + head_size]
                                .copy_from_slice(&s.hb[src_base..src_base + head_size]);
                        }
                    }
                    qwen3next_packed_q_gate = true;
                } else if w.wq[l].rows == q_dim {
                    matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.q[..q_dim],
                        &s.xb[..dim],
                        &w.wq[l],
                        0,
                        q_dim,
                        mapped,
                    )?;
                }
                if w.wk[l].rows == kv_dim {
                    matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.k[..kv_dim],
                        &s.xb[..dim],
                        &w.wk[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
                if w.wv[l].rows == kv_dim {
                    matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
                } else {
                    matmul_quantized_rows(
                        &mut s.v[..kv_dim],
                        &s.xb[..dim],
                        &w.wv[l],
                        0,
                        kv_dim,
                        mapped,
                    )?;
                }
            } else {
                matmul_quantized(&mut s.q[..q_dim], &s.xb[..dim], &w.wq[l], mapped)?;
                matmul_quantized(&mut s.k[..kv_dim], &s.xb[..dim], &w.wk[l], mapped)?;
                matmul_quantized(&mut s.v[..kv_dim], &s.xb[..dim], &w.wv[l], mapped)?;
            }

            if p.is_qwen2 && !w.attn_q_bias.is_empty() {
                let qb = &w.attn_q_bias[l * q_dim..(l + 1) * q_dim];
                let kb = &w.attn_k_bias[l * kv_dim..(l + 1) * kv_dim];
                let vb = &w.attn_v_bias[l * kv_dim..(l + 1) * kv_dim];
                for i in 0..q_dim {
                    s.q[i] += qb[i];
                }
                for i in 0..kv_dim {
                    s.k[i] += kb[i];
                    s.v[i] += vb[i];
                }
            }

            if !w.attn_q_norm.is_empty()
                && !w.attn_k_norm.is_empty()
                && !w.attn_qk_norm_present.is_empty()
                && w.attn_qk_norm_present[l]
            {
                let q_norm = &w.attn_q_norm[l * head_size..(l + 1) * head_size];
                let k_norm = &w.attn_k_norm[l * head_size..(l + 1) * head_size];
                rmsnorm_per_head_gemma_inplace(
                    &mut s.q[..q_dim],
                    q_norm,
                    p.n_heads,
                    head_size,
                    eps,
                );
                rmsnorm_per_head_gemma_inplace(
                    &mut s.k[..kv_dim],
                    k_norm,
                    p.n_kv_heads,
                    head_size,
                    eps,
                );
            }

            let is_swa_layer = p.swa_pattern > 0 && (l % p.swa_pattern < p.swa_pattern - 1);
            let rope_freqs = if p.is_gemma3 && is_swa_layer {
                &s.rope_freqs_swa
            } else {
                &s.rope_freqs
            };
            let rope_half = s.rope_cos.len();
            let current_is_swa = if is_swa_layer { 1 } else { 0 };
            if s.rope_cache_pos != pos as isize || s.rope_cache_is_swa != current_is_swa {
                for i in 0..rope_half {
                    let val = pos as f32 * rope_freqs[i];
                    s.rope_cos[i] = val.cos();
                    s.rope_sin[i] = val.sin();
                }
                s.rope_cache_pos = pos as isize;
                s.rope_cache_is_swa = current_is_swa;
            }

            if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
                let pair_offset = rope_half;
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.q[hs + i];
                        let v1 = s.q[hs + i + pair_offset];
                        s.q[hs + i] = v0 * fcr - v1 * fci;
                        s.q[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
                for h in 0..p.n_kv_heads {
                    let hs = h * head_size;
                    for i in 0..rope_half {
                        let fcr = s.rope_cos[i];
                        let fci = s.rope_sin[i];
                        let v0 = s.k[hs + i];
                        let v1 = s.k[hs + i + pair_offset];
                        s.k[hs + i] = v0 * fcr - v1 * fci;
                        s.k[hs + i + pair_offset] = v0 * fci + v1 * fcr;
                    }
                }
            } else {
                let mut i = 0;
                while i < q_dim {
                    let head_dim_idx = (i % head_size) / 2;
                    let fcr = s.rope_cos[head_dim_idx];
                    let fci = s.rope_sin[head_dim_idx];
                    let rotn = if i < kv_dim { 2 } else { 1 };
                    for v in 0..rotn {
                        let vec = if v == 0 { &mut s.q } else { &mut s.k };
                        let v0 = vec[i];
                        let v1 = vec[i + 1];
                        vec[i] = v0 * fcr - v1 * fci;
                        vec[i + 1] = v0 * fci + v1 * fcr;
                    }
                    i += 2;
                }
            }

            if p.is_gemma3 {
                scale_slice_inplace(&mut s.q[..q_dim], s.attn_scale);
            }

            let loff = l * s.kv_cache_layer_size;
            let key_cache_row = loff + pos * kv_dim;
            let value_cache_row = loff + pos * kv_dim;
            s.key_cache[key_cache_row..key_cache_row + kv_dim].copy_from_slice(&s.k[..kv_dim]);
            s.value_cache[value_cache_row..value_cache_row + kv_dim]
                .copy_from_slice(&s.v[..kv_dim]);

            let attn_scale_score = s.attn_scale;
            let apply_attn_scale = !p.is_gemma3;
            let q_all = &s.q[..q_dim];
            let key_cache = &s.key_cache;
            let value_cache = &s.value_cache;
            let (att_all, xb_all) = (&mut s.att[..p.n_heads * p.seq_len], &mut s.xb[..q_dim]);

            if p.n_heads >= par_attn_min_heads() {
                att_all
                    .par_chunks_mut(p.seq_len)
                    .zip(xb_all.par_chunks_mut(head_size))
                    .enumerate()
                    .for_each(|(h, (att_head_full, xb_head))| {
                        let hs = h * head_size;
                        let q_head = &q_all[hs..hs + head_size];
                        let kv_head = h / kv_mul;
                        let kv_head_offset = kv_head * head_size;

                        let att_head = &mut att_head_full[..=pos];
                        for (t, slot) in att_head.iter_mut().enumerate() {
                            let k_off = loff + t * kv_dim + kv_head_offset;
                            let k_head = &key_cache[k_off..k_off + head_size];
                            let mut score = dot_f32_simd(q_head, k_head);
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }
                            *slot = score;
                        }

                        softmax(att_head, pos + 1);

                        xb_head.fill(0.0);
                        for (t, &a) in att_head.iter().enumerate() {
                            let v_off = loff + t * kv_dim + kv_head_offset;
                            let v_head = &value_cache[v_off..v_off + head_size];
                            axpy_inplace(xb_head, a, v_head);
                        }
                    });
            } else {
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    let q_head = &q_all[hs..hs + head_size];
                    let kv_head = h / kv_mul;
                    let kv_head_offset = kv_head * head_size;
                    let att_head_full = &mut att_all[h * p.seq_len..(h + 1) * p.seq_len];
                    let att_head = &mut att_head_full[..=pos];

                    for (t, slot) in att_head.iter_mut().enumerate() {
                        let k_off = loff + t * kv_dim + kv_head_offset;
                        let k_head = &key_cache[k_off..k_off + head_size];
                        let mut score = dot_f32_simd(q_head, k_head);
                        if apply_attn_scale {
                            score *= attn_scale_score;
                        }
                        *slot = score;
                    }

                    softmax(att_head, pos + 1);

                    let xb_head = &mut xb_all[hs..hs + head_size];
                    xb_head.fill(0.0);
                    for (t, &a) in att_head.iter().enumerate() {
                        let v_off = loff + t * kv_dim + kv_head_offset;
                        let v_head = &value_cache[v_off..v_off + head_size];
                        axpy_inplace(xb_head, a, v_head);
                    }
                }
            }

            if qwen3next_packed_q_gate {
                for h in 0..p.n_heads {
                    let src_base = h * 2 * head_size + head_size;
                    let dst_base = h * head_size;
                    for i in 0..head_size {
                        s.xb[dst_base + i] =
                            finite_or_zero(s.xb[dst_base + i] * sigmoidf(s.hb[src_base + i]));
                    }
                }
            }
            matmul_quantized(&mut s.xb2[..dim], &s.xb[..q_dim], &w.wo[l], mapped)?;
            for v in &mut s.xb2[..dim] {
                *v = finite_or_zero(*v);
            }
            prof_end(&PROF_ATTN_NS, attn_prof);
        }

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_attn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb2[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }

        if p.is_gemma3 && !w.attn_post_norm.is_empty() {
            rmsnorm_inplace(
                &mut s.xb2[..dim],
                &w.attn_post_norm[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        accum(&mut s.x[..dim], &s.xb2[..dim], dim);

        if p.is_gemma3 {
            rmsnorm_gemma(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        } else {
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[..dim],
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        if p.is_qwen3moe || p.is_qwen3next {
            let moe_prof = prof_start();
            let expert_hidden = p.expert_hidden_dim;
            s.xb2[..dim].copy_from_slice(&s.xb[..dim]);
            matmul_quantized(
                &mut s.moe_logits[..p.n_experts],
                &s.xb2[..dim],
                &w.moe_gate_inp[l],
                mapped,
            )?;
            let n_selected = select_topk_softmax(
                &s.moe_logits[..p.n_experts],
                p.n_experts_used,
                p.moe_n_group,
                p.moe_topk_group,
                p.moe_norm_topk_prob,
                p.moe_routed_scaling_factor,
                &mut s.moe_scores,
                &mut s.moe_selected_group,
                &mut s.moe_group_scores,
                &mut s.moe_group_rank,
                &mut s.moe_topk_indices,
                &mut s.moe_topk_weights,
            );
            s.xb[..dim].fill(0.0);

            for j in 0..n_selected {
                let expert_idx = s.moe_topk_indices[j];
                let route_weight = s.moe_topk_weights[j];
                if route_weight == 0.0 {
                    continue;
                }

                let row_start_ffn = expert_idx * expert_hidden;
                matmul_quantized_rows(
                    &mut s.hb[..expert_hidden],
                    &s.xb2[..dim],
                    &w.moe_gate_exps[l],
                    row_start_ffn,
                    expert_hidden,
                    mapped,
                )?;
                matmul_quantized_rows(
                    &mut s.hb2[..expert_hidden],
                    &s.xb2[..dim],
                    &w.moe_up_exps[l],
                    row_start_ffn,
                    expert_hidden,
                    mapped,
                )?;

                for i in 0..expert_hidden {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }

                let row_start_down = expert_idx * dim;
                matmul_quantized_rows(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..expert_hidden],
                    &w.moe_down_exps[l],
                    row_start_down,
                    dim,
                    mapped,
                )?;
                for i in 0..dim {
                    s.xb[i] += route_weight * s.moe_tmp[i];
                }
            }

            if p.is_qwen3next && !w.moe_shared_gate_inp.is_empty() {
                let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                    p.shared_expert_hidden_dim
                } else {
                    p.expert_hidden_dim
                };
                let shared_gate = &w.moe_shared_gate_inp[l * dim..(l + 1) * dim];
                let gate_logit = dot_f32_simd(&s.xb2[..dim], shared_gate);
                let gate = 1.0 / (1.0 + (-gate_logit).exp());

                matmul_quantized(&mut s.hb[..shared_hidden], &s.xb2[..dim], &w.w1[l], mapped)?;
                matmul_quantized(&mut s.hb2[..shared_hidden], &s.xb2[..dim], &w.w3[l], mapped)?;
                for i in 0..shared_hidden {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }
                matmul_quantized(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..shared_hidden],
                    &w.w2[l],
                    mapped,
                )?;
                for i in 0..dim {
                    s.xb[i] += gate * s.moe_tmp[i];
                }
            }
            prof_end(&PROF_MOE_NS, moe_prof);
        } else {
            let ffn_prof = prof_start();
            matmul_quantized(&mut s.hb[..hidden_dim], &s.xb[..dim], &w.w1[l], mapped)?;
            matmul_quantized(&mut s.hb2[..hidden_dim], &s.xb[..dim], &w.w3[l], mapped)?;

            if p.is_gemma3 {
                for i in 0..hidden_dim {
                    let x = s.hb[i];
                    let gelu =
                        0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh());
                    s.hb[i] = gelu * s.hb2[i];
                }
            } else {
                for i in 0..hidden_dim {
                    let v = s.hb[i];
                    s.hb[i] = (v * (1.0 / (1.0 + (-v).exp()))) * s.hb2[i];
                }
            }

            matmul_quantized(&mut s.xb[..dim], &s.hb[..hidden_dim], &w.w2[l], mapped)?;
            prof_end(&PROF_FFN_NS, ffn_prof);
        }

        if p.is_gemma3 && !w.ffn_post_norm.is_empty() {
            rmsnorm_inplace(
                &mut s.xb[..dim],
                &w.ffn_post_norm[l * dim..(l + 1) * dim],
                dim,
                eps,
            );
        }

        accum(&mut s.x[..dim], &s.xb[..dim], dim);

        if do_layer_debug {
            eprintln!(
                "[LAYERDBG pos={pos} l={l}] post_ffn_norm={:.4} x_norm={:.4}",
                l2_norm(&s.xb[..dim]),
                l2_norm(&s.x[..dim]),
            );
        }
    }

    if p.is_gemma3 {
        rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    } else {
        rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    }
    for v in &mut s.x[..dim] {
        *v = finite_or_zero(*v);
    }

    if w.wcls_is_embed {
        matmul_f32_embeddings(
            &mut s.logits[..p.vocab_size],
            &s.x[..dim],
            &w.token_embedding_table,
            p.vocab_size,
            dim,
        );
    } else {
        matmul_quantized(&mut s.logits[..p.vocab_size], &s.x[..dim], &w.wcls, mapped)?;
    }
    for v in &mut s.logits[..p.vocab_size] {
        *v = finite_or_zero(*v);
    }

    if p.is_gemma3 && p.final_logit_softcapping > 0.0 {
        let cap = p.final_logit_softcapping;
        for i in 0..p.vocab_size {
            s.logits[i] = cap * (s.logits[i] / cap).tanh();
        }
    }

    Ok(())
}
