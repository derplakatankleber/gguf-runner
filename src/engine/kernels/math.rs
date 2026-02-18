#![allow(clippy::needless_range_loop)]

use crate::engine::kernels::{
    axpy_inplace, dot_f32_simd, matmul_quantized, matmul_quantized_rows, scale_slice_inplace,
};
use crate::engine::profiling::{prof_end, prof_start, PROF_SSM_NS};
use crate::engine::switches::par_qwen3next_min_heads;
use crate::engine::types::{Config, RunState, TransformerWeights};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};
pub(crate) fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}

pub(crate) fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32) {
    let mut ss = 0.0f32;
    for i in 0..size {
        ss += x[i] * x[i];
    }
    ss /= size as f32;
    ss += eps;
    let ss = 1.0 / ss.sqrt();
    for i in 0..size {
        o[i] = weight[i] * (ss * x[i]);
    }
}

pub(crate) fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], size: usize, eps: f32) {
    let mut ss = 0.0f32;
    for i in 0..size {
        ss += x[i] * x[i];
    }
    ss /= size as f32;
    ss += eps;
    let ss = 1.0 / ss.sqrt();
    for i in 0..size {
        x[i] = weight[i] * (ss * x[i]);
    }
}

pub(crate) fn rmsnorm_gemma(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32) {
    let mut ss = 0.0f32;
    for i in 0..size {
        ss += x[i] * x[i];
    }
    ss /= size as f32;
    ss += eps;
    let ss = 1.0 / ss.sqrt();
    for i in 0..size {
        o[i] = weight[i] * (ss * x[i]);
    }
}

pub(crate) fn rmsnorm_per_head_gemma_inplace(
    x: &mut [f32],
    weight: &[f32],
    n_heads: usize,
    head_size: usize,
    eps: f32,
) {
    for h in 0..n_heads {
        let hs = h * head_size;
        let mut ss = 0.0f32;
        for j in 0..head_size {
            ss += x[hs + j] * x[hs + j];
        }
        ss /= head_size as f32;
        ss += eps;
        let ss = 1.0 / ss.sqrt();
        for j in 0..head_size {
            x[hs + j] = weight[j] * (ss * x[hs + j]);
        }
    }
}

pub(crate) fn softmax(x: &mut [f32], size: usize) {
    let mut max_val = x[0];
    for &v in x.iter().take(size).skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0f32;
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }

    let inv_sum = 1.0 / sum;
    for i in 0..size {
        x[i] *= inv_sum;
    }
}

#[inline(always)]
pub(crate) fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub(crate) fn siluf(x: f32) -> f32 {
    x * sigmoidf(x)
}

#[inline(always)]
pub(crate) fn softplusf(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
pub(crate) fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[inline(always)]
pub(crate) fn l2_norm(x: &[f32]) -> f32 {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss.sqrt()
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen3next_state_head_step(
    state_h: &mut [f32],
    out_h: &mut [f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
) {
    let head_dim = q.len();
    debug_assert_eq!(k.len(), head_dim);
    debug_assert_eq!(v.len(), head_dim);
    debug_assert_eq!(out_h.len(), head_dim);
    debug_assert_eq!(kv_mem.len(), head_dim);
    debug_assert_eq!(delta.len(), head_dim);
    debug_assert_eq!(state_h.len(), head_dim * head_dim);

    scale_slice_inplace(state_h, g);

    kv_mem.fill(0.0);
    for j in 0..head_dim {
        let kj = k[j];
        if kj == 0.0 {
            continue;
        }
        let col = &state_h[j * head_dim..(j + 1) * head_dim];
        axpy_inplace(kv_mem, kj, col);
    }
    for i in 0..head_dim {
        kv_mem[i] = finite_or_zero(kv_mem[i]);
        delta[i] = finite_or_zero((v[i] - kv_mem[i]) * beta);
    }

    for j in 0..head_dim {
        let kj = k[j];
        if kj == 0.0 {
            continue;
        }
        let col = &mut state_h[j * head_dim..(j + 1) * head_dim];
        axpy_inplace(col, kj, delta);
    }

    out_h.fill(0.0);
    for j in 0..head_dim {
        let qj = q[j];
        if qj == 0.0 {
            continue;
        }
        let col = &state_h[j * head_dim..(j + 1) * head_dim];
        axpy_inplace(out_h, qj, col);
    }
    for v in out_h {
        *v = finite_or_zero(*v);
    }
}

pub(crate) fn qwen3next_linear_attention_autoregressive(
    l: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
    eps: f32,
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d_inner = p.ssm_inner_size;
    let n_k_heads = p.ssm_group_count;
    let n_v_heads = p.ssm_time_step_rank;
    let head_dim = p.ssm_state_size;
    let conv_kernel = p.ssm_conv_kernel;
    let conv_dim = d_inner + 2 * n_k_heads * head_dim;

    if d_inner == 0 || n_k_heads == 0 || n_v_heads == 0 || head_dim == 0 || conv_kernel == 0 {
        return Err("invalid qwen3next SSM config".to_string());
    }
    if !n_v_heads.is_multiple_of(n_k_heads) {
        return Err(format!(
            "unsupported qwen3next SSM shape: n_v_heads {} not divisible by n_k_heads {}",
            n_v_heads, n_k_heads
        ));
    }
    if head_dim * n_v_heads != d_inner {
        return Err(format!(
            "unsupported qwen3next SSM shape: head_dim*n_v_heads {} != d_inner {}",
            head_dim * n_v_heads,
            d_inner
        ));
    }
    if w.ssm_conv1d.is_empty()
        || w.ssm_ba.is_empty()
        || w.ssm_a.is_empty()
        || w.ssm_dt_bias.is_empty()
        || w.ssm_norm.is_empty()
    {
        return Err("missing qwen3next SSM tensors".to_string());
    }
    if l >= w.ssm_conv1d.len() || l >= w.ssm_ba.len() {
        return Err("qwen3next SSM layer index out of range".to_string());
    }
    if w.attn_qkv[l].rows < conv_dim {
        return Err(format!(
            "blk.{l}.attn_qkv.weight has {} rows, expected at least {}",
            w.attn_qkv[l].rows, conv_dim
        ));
    }
    if w.wo[l].rows < d_inner {
        return Err(format!(
            "blk.{l}.attn_gate.weight has {} rows, expected at least {}",
            w.wo[l].rows, d_inner
        ));
    }
    if w.ssm_ba[l].rows < 2 * n_v_heads {
        return Err(format!(
            "blk.{l}.ssm_ba.weight has {} rows, expected at least {}",
            w.ssm_ba[l].rows,
            2 * n_v_heads
        ));
    }

    matmul_quantized_rows(
        &mut s.ssm_qkv[..conv_dim],
        &s.xb[..p.dim],
        &w.attn_qkv[l],
        0,
        conv_dim,
        mapped,
    )?;
    matmul_quantized_rows(
        &mut s.ssm_z[..d_inner],
        &s.xb[..p.dim],
        &w.wo[l],
        0,
        d_inner,
        mapped,
    )?;
    matmul_quantized_rows(
        &mut s.ssm_ba[..2 * n_v_heads],
        &s.xb[..p.dim],
        &w.ssm_ba[l],
        0,
        2 * n_v_heads,
        mapped,
    )?;

    let conv_w = &w.ssm_conv1d[l];
    if conv_w.len() < conv_kernel * conv_dim {
        return Err(format!(
            "blk.{l}.ssm_conv1d.weight has {} elements, expected at least {}",
            conv_w.len(),
            conv_kernel * conv_dim
        ));
    }

    let hist_steps = conv_kernel - 1;
    let conv_hist_stride = hist_steps * conv_dim;
    let conv_hist_off = l * conv_hist_stride;
    if conv_hist_off + conv_hist_stride > s.ssm_conv_state.len() {
        return Err("qwen3next conv state buffer too small".to_string());
    }
    let conv_hist = &mut s.ssm_conv_state[conv_hist_off..conv_hist_off + conv_hist_stride];

    for c in 0..conv_dim {
        let mut acc = s.ssm_qkv[c] * conv_w[c * conv_kernel + hist_steps];
        for t in 0..hist_steps {
            acc += conv_hist[t * conv_dim + c] * conv_w[c * conv_kernel + t];
        }
        if !acc.is_finite() {
            acc = 0.0;
        }
        s.ssm_conv[c] = siluf(acc);
    }

    if hist_steps > 0 {
        if hist_steps > 1 {
            conv_hist.copy_within(conv_dim.., 0);
        }
        let tail = (hist_steps - 1) * conv_dim;
        conv_hist[tail..tail + conv_dim].copy_from_slice(&s.ssm_qkv[..conv_dim]);
    }

    let q_off = 0usize;
    let k_off = n_k_heads * head_dim;
    let v_off = 2 * n_k_heads * head_dim;
    let repeat = n_v_heads / n_k_heads;
    let inv_scale_q = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_v_heads {
        let src_h = h / repeat;
        let q_src = &s.ssm_conv[q_off + src_h * head_dim..q_off + (src_h + 1) * head_dim];
        let k_src = &s.ssm_conv[k_off + src_h * head_dim..k_off + (src_h + 1) * head_dim];
        let v_src = &s.ssm_conv[v_off + h * head_dim..v_off + (h + 1) * head_dim];

        let q_dst = &mut s.ssm_q[h * head_dim..(h + 1) * head_dim];
        let k_dst = &mut s.ssm_k[h * head_dim..(h + 1) * head_dim];
        let v_dst = &mut s.ssm_v[h * head_dim..(h + 1) * head_dim];

        let mut q_ss = 0.0f32;
        let mut k_ss = 0.0f32;
        for i in 0..head_dim {
            q_ss += q_src[i] * q_src[i];
            k_ss += k_src[i] * k_src[i];
        }
        let q_inv = 1.0 / (q_ss + eps).sqrt();
        let k_inv = 1.0 / (k_ss + eps).sqrt();
        for i in 0..head_dim {
            q_dst[i] = finite_or_zero(q_src[i] * q_inv * inv_scale_q);
            k_dst[i] = finite_or_zero(k_src[i] * k_inv);
            v_dst[i] = finite_or_zero(v_src[i]);
        }
    }

    let heads_per_group = n_v_heads / n_k_heads;
    let dt_base = l * n_v_heads;
    let a_base = l * n_v_heads;
    for h in 0..n_v_heads {
        let group = h / heads_per_group;
        let idx = h % heads_per_group;
        let base = group * (2 * heads_per_group);
        let beta = sigmoidf(s.ssm_ba[base + idx]);
        let alpha = s.ssm_ba[base + heads_per_group + idx] + w.ssm_dt_bias[dt_base + h];
        let mut gate = softplusf(alpha) * w.ssm_a[a_base + h];
        if !gate.is_finite() {
            gate = 0.0;
        }
        s.ssm_beta[h] = finite_or_zero(beta);
        s.ssm_gate_exp[h] = finite_or_zero(gate.exp());
    }
    let state_stride = n_v_heads * head_dim * head_dim;
    let state_off = l * state_stride;
    if state_off + state_stride > s.ssm_state.len() {
        return Err("qwen3next state buffer too small".to_string());
    }
    let state = &mut s.ssm_state[state_off..state_off + state_stride];
    let q_all = &s.ssm_q[..d_inner];
    let k_all = &s.ssm_k[..d_inner];
    let v_all = &s.ssm_v[..d_inner];
    let gate_all = &s.ssm_gate_exp[..n_v_heads];
    let beta_all = &s.ssm_beta[..n_v_heads];
    let proj_all = &mut s.ssm_proj[..d_inner];
    let kv_mem_all = &mut s.ssm_kv_mem[..d_inner];
    let delta_all = &mut s.ssm_delta[..d_inner];

    if n_v_heads >= par_qwen3next_min_heads() {
        state
            .par_chunks_mut(head_dim * head_dim)
            .zip(proj_all.par_chunks_mut(head_dim))
            .zip(kv_mem_all.par_chunks_mut(head_dim))
            .zip(delta_all.par_chunks_mut(head_dim))
            .enumerate()
            .for_each(|(h, (((state_h, out_h), kv_mem), delta))| {
                let q = &q_all[h * head_dim..(h + 1) * head_dim];
                let k = &k_all[h * head_dim..(h + 1) * head_dim];
                let v = &v_all[h * head_dim..(h + 1) * head_dim];
                qwen3next_state_head_step(
                    state_h,
                    out_h,
                    kv_mem,
                    delta,
                    q,
                    k,
                    v,
                    gate_all[h],
                    beta_all[h],
                );
            });
    } else {
        for h in 0..n_v_heads {
            let q = &q_all[h * head_dim..(h + 1) * head_dim];
            let k = &k_all[h * head_dim..(h + 1) * head_dim];
            let v = &v_all[h * head_dim..(h + 1) * head_dim];
            let state_h = &mut state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
            let out_h = &mut proj_all[h * head_dim..(h + 1) * head_dim];
            let kv_mem = &mut kv_mem_all[h * head_dim..(h + 1) * head_dim];
            let delta = &mut delta_all[h * head_dim..(h + 1) * head_dim];
            qwen3next_state_head_step(
                state_h,
                out_h,
                kv_mem,
                delta,
                q,
                k,
                v,
                gate_all[h],
                beta_all[h],
            );
        }
    }

    let ssm_norm = &w.ssm_norm[l * head_dim..(l + 1) * head_dim];
    if n_v_heads >= par_qwen3next_min_heads() {
        proj_all
            .par_chunks_mut(head_dim)
            .zip(s.ssm_z[..d_inner].par_chunks(head_dim))
            .for_each(|(out_h, z_h)| {
                let mut ss = 0.0f32;
                for i in 0..head_dim {
                    ss += out_h[i] * out_h[i];
                }
                let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
                for i in 0..head_dim {
                    out_h[i] = finite_or_zero(ssm_norm[i] * (out_h[i] * inv) * siluf(z_h[i]));
                }
            });
    } else {
        for h in 0..n_v_heads {
            let out_h = &mut proj_all[h * head_dim..(h + 1) * head_dim];
            let z_h = &s.ssm_z[h * head_dim..(h + 1) * head_dim];
            let mut ss = 0.0f32;
            for i in 0..head_dim {
                ss += out_h[i] * out_h[i];
            }
            let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
            for i in 0..head_dim {
                out_h[i] = finite_or_zero(ssm_norm[i] * (out_h[i] * inv) * siluf(z_h[i]));
            }
        }
    }

    matmul_quantized(
        &mut s.xb2[..p.dim],
        &s.ssm_proj[..d_inner],
        &w.wv[l],
        mapped,
    )?;
    for v in &mut s.xb2[..p.dim] {
        *v = finite_or_zero(*v);
    }
    prof_end(&PROF_SSM_NS, prof_t0);
    Ok(())
}

pub(crate) fn matmul_f32_embeddings(
    logits: &mut [f32],
    x: &[f32],
    emb: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row = &emb[r * cols..(r + 1) * cols];
        logits[r] = dot_f32_simd(row, &x[..cols]);
    }
}
