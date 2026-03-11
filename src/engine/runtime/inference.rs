use crate::engine::kernels::{
    accum, dot_f32_simd, get_row_size, l2_norm, matmul_f32_embeddings, matmul_quantized,
    matmul_quantized_rows, qwen3next_linear_attention_autoregressive, rmsnorm, rmsnorm_gemma,
    rmsnorm_inplace, rmsnorm_per_head_gemma_inplace, sanitize_finite_inplace, scale_slice_inplace,
    select_topk_softmax, sigmoid_mul_inplace, silu_and_mul_inplace, softmax,
};
use crate::engine::profiling::{prof_end, prof_start, PROF_ATTN_NS, PROF_FFN_NS, PROF_MOE_NS};
use crate::engine::switches::{
    kv_cache_mode, layer_debug_enabled, layer_debug_pos, par_attn_min_heads,
    KvCacheMode as SwitchKvCacheMode,
};
use crate::engine::types::{
    Config, KvCacheFormat, QuantizedTensor, RunState, TransformerWeights, GGML_TYPE_BF16,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

#[inline]
fn bf16_le_to_f32(lo: u8, hi: u8) -> f32 {
    let u = u16::from_le_bytes([lo, hi]) as u32;
    f32::from_bits(u << 16)
}

#[allow(clippy::too_many_arguments)]
fn validate_bf16_projection_rows(
    p: &Config,
    pos: usize,
    l: usize,
    tag: &str,
    qw: &QuantizedTensor,
    x: &[f32],
    got: &[f32],
    mapped: &[u8],
) {
    if !p.is_qwen35 || qw.ttype.0 != GGML_TYPE_BF16 || !env_flag("GGUF_QWEN35_VALIDATE_BF16_KV") {
        return;
    }
    let target_layer = env_usize("GGUF_QWEN35_VALIDATE_LAYER").unwrap_or(3);
    let target_pos = env_usize("GGUF_QWEN35_VALIDATE_POS").unwrap_or(23);
    if l != target_layer || pos != target_pos {
        return;
    }
    let row_size = get_row_size(qw.cols, qw.ttype);
    if row_size != qw.cols * 2 {
        eprintln!(
            "[BF16CHK] unexpected row_size for {tag}: row_size={} cols={} ttype={}",
            row_size, qw.cols, qw.ttype.0
        );
        return;
    }

    let probes = [0usize, 1, 2, 7, 15, 31, 63, 127, 255, 511];
    let mut max_abs = 0.0f32;
    let mut worst_row = 0usize;
    let mut samples = 0usize;
    for &r in &probes {
        if r >= qw.rows || r >= got.len() {
            continue;
        }
        let row_off = match qw.data_offset.checked_add(r.saturating_mul(row_size)) {
            Some(v) => v,
            None => continue,
        };
        let row_end = match row_off.checked_add(row_size) {
            Some(v) => v,
            None => continue,
        };
        if row_end > mapped.len() {
            continue;
        }
        let row = &mapped[row_off..row_end];
        let mut ref_dot = 0.0f32;
        for (i, &xv) in x.iter().enumerate().take(qw.cols) {
            let b = i * 2;
            ref_dot += xv * bf16_le_to_f32(row[b], row[b + 1]);
        }
        let diff = (got[r] - ref_dot).abs();
        if diff > max_abs {
            max_abs = diff;
            worst_row = r;
        }
        samples += 1;
    }

    if samples > 0 {
        eprintln!(
            "[BF16CHK] pos={} layer={} tensor={} samples={} max_abs_diff={:.6e} worst_row={}",
            pos, l, tag, samples, max_abs, worst_row
        );
    }
}

fn alloc_f32(len: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        let bytes = len.saturating_mul(std::mem::size_of::<f32>());
        format!(
            "unable to allocate {label} ({bytes} bytes). Try reducing --context-size and --max-tokens."
        )
    })?;
    out.resize(len, 0.0);
    Ok(out)
}

fn alloc_i8(len: usize, label: &str) -> Result<Vec<i8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn alloc_u8(len: usize, label: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        format!(
            "unable to allocate {label} ({} bytes). Try reducing --context-size and --max-tokens.",
            len
        )
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn quantize_row_q8(src: &[f32], dst: &mut [i8], scale_out: &mut f32) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = src.len();
        // Vectorized abs-max scan
        let max_abs = unsafe {
            let mut vmax = vdupq_n_f32(0.0f32);
            let mut i = 0usize;
            while i + 16 <= n {
                let v0 = vabsq_f32(vld1q_f32(src.as_ptr().add(i)));
                let v1 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 4)));
                let v2 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 8)));
                let v3 = vabsq_f32(vld1q_f32(src.as_ptr().add(i + 12)));
                vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
                i += 16;
            }
            while i + 4 <= n {
                vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(src.as_ptr().add(i))));
                i += 4;
            }
            let mut m = vmaxvq_f32(vmax);
            while i < n {
                let a = src[i].abs();
                if a > m {
                    m = a;
                }
                i += 1;
            }
            m
        };
        if max_abs == 0.0 {
            *scale_out = 1.0;
            dst.fill(0);
            return;
        }
        *scale_out = max_abs / 127.0;
        let inv = 127.0 / max_abs;
        // Vectorized quantize
        unsafe {
            let vinv = vdupq_n_f32(inv);
            let vmin = vdupq_n_f32(-127.0);
            let vmax = vdupq_n_f32(127.0);
            let mut i = 0usize;
            let dst_ptr = dst.as_mut_ptr();
            while i + 16 <= n {
                let f0 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i)), vinv)),
                    ),
                );
                let f1 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 4)), vinv)),
                    ),
                );
                let f2 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 8)), vinv)),
                    ),
                );
                let f3 = vmaxq_f32(
                    vmin,
                    vminq_f32(
                        vmax,
                        vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i + 12)), vinv)),
                    ),
                );
                let i0 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(f0)), vmovn_s32(vcvtq_s32_f32(f1)));
                let i1 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(f2)), vmovn_s32(vcvtq_s32_f32(f3)));
                vst1q_s8(dst_ptr.add(i), vcombine_s8(vmovn_s16(i0), vmovn_s16(i1)));
                i += 16;
            }
            while i < n {
                dst[i] = (src[i] * inv).round().clamp(-127.0, 127.0) as i8;
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut max_abs = 0.0f32;
        for &x in src {
            max_abs = max_abs.max(x.abs());
        }
        if max_abs == 0.0 {
            *scale_out = 1.0;
            dst.fill(0);
            return;
        }
        let inv = 127.0 / max_abs;
        *scale_out = max_abs / 127.0;
        for (i, &x) in src.iter().enumerate() {
            dst[i] = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q8_row_neon(q: &[f32], cache: &[i8], row_offset: usize, scale: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = q.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let q_ptr = q.as_ptr();
    let k_ptr = cache.as_ptr().add(row_offset);
    let mut i = 0usize;
    while i + 16 <= n {
        let kv = vld1q_s8(k_ptr.add(i));
        let k_lo = vmovl_s8(vget_low_s8(kv));
        let k_hi = vmovl_s8(vget_high_s8(kv));
        let k0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_lo)));
        let k1 = vcvtq_f32_s32(vmovl_high_s16(k_lo));
        let k2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_hi)));
        let k3 = vcvtq_f32_s32(vmovl_high_s16(k_hi));
        acc0 = vfmaq_f32(acc0, vld1q_f32(q_ptr.add(i)), k0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(q_ptr.add(i + 4)), k1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(q_ptr.add(i + 8)), k2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(q_ptr.add(i + 12)), k3);
        i += 16;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        sum += q[i] * cache[row_offset + i] as f32;
        i += 1;
    }
    sum * scale
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_q8_row_neon(dst: &mut [f32], a: f32, cache: &[i8], row_offset: usize, scale: f32) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let scaled = vdupq_n_f32(a * scale);
    let dst_ptr = dst.as_mut_ptr();
    let k_ptr = cache.as_ptr().add(row_offset);
    let mut i = 0usize;
    while i + 16 <= n {
        let kv = vld1q_s8(k_ptr.add(i));
        let k_lo = vmovl_s8(vget_low_s8(kv));
        let k_hi = vmovl_s8(vget_high_s8(kv));
        let k0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_lo)));
        let k1 = vcvtq_f32_s32(vmovl_high_s16(k_lo));
        let k2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_hi)));
        let k3 = vcvtq_f32_s32(vmovl_high_s16(k_hi));
        vst1q_f32(
            dst_ptr.add(i),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i)), scaled, k0),
        );
        vst1q_f32(
            dst_ptr.add(i + 4),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 4)), scaled, k1),
        );
        vst1q_f32(
            dst_ptr.add(i + 8),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 8)), scaled, k2),
        );
        vst1q_f32(
            dst_ptr.add(i + 12),
            vfmaq_f32(vld1q_f32(dst_ptr.add(i + 12)), scaled, k3),
        );
        i += 16;
    }
    let scalar = a * scale;
    while i < n {
        dst[i] += scalar * cache[row_offset + i] as f32;
        i += 1;
    }
}

/// Block size for Q4 KV cache quantization. Each block gets its own scale factor,
/// matching the Q4_0 standard. Smaller blocks preserve more precision by limiting
/// the impact of outlier activations.
const Q4_BLOCK_SIZE: usize = 32;

fn quantize_q4_block(src: &[f32], dst: &mut [u8], base_elem: usize, scale_out: &mut f32) {
    debug_assert_eq!(src.len(), Q4_BLOCK_SIZE);
    let mut max_abs = 0.0f32;
    for &x in src {
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        *scale_out = 1.0;
        for i in 0..src.len() {
            let elem_idx = base_elem + i;
            let byte_idx = elem_idx / 2;
            if (elem_idx & 1) == 0 {
                dst[byte_idx] &= 0xF0;
            } else {
                dst[byte_idx] &= 0x0F;
            }
        }
        return;
    }
    let inv = 7.0 / max_abs;
    let scale = max_abs / 7.0;
    *scale_out = scale;
    for (i, &x) in src.iter().enumerate() {
        let q = (x * inv).round().clamp(-8.0, 7.0) as i8;
        let nib = (q as i32 & 0x0F) as u8;
        let elem_idx = base_elem + i;
        let byte_idx = elem_idx / 2;
        if (elem_idx & 1) == 0 {
            dst[byte_idx] = (dst[byte_idx] & 0xF0) | nib;
        } else {
            dst[byte_idx] = (dst[byte_idx] & 0x0F) | (nib << 4);
        }
    }
}

#[inline]
fn dequant_q4_at(src: &[u8], elem_idx: usize) -> i8 {
    let byte = src[elem_idx / 2];
    let nib = if (elem_idx & 1) == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    if nib >= 8 {
        nib as i8 - 16
    } else {
        nib as i8
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn s8x8_to_f32x4x2(
    v: std::arch::aarch64::int8x8_t,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let s16 = vmovl_s8(v);
    let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16)));
    let hi = vcvtq_f32_s32(vmovl_high_s16(s16));
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q4_block_neon(q_block: &[f32], cache: &[u8], elem_offset: usize) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(q_block.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(elem_offset & 1, 0);

    let packed_ptr = cache.as_ptr().add(elem_offset / 2);
    let q_ptr = q_block.as_ptr();
    let nib_mask = vdup_n_u8(0x0f);
    let sign_xor = vdup_n_u8(0x08);
    let sign_sub = vdup_n_s8(8);
    let mut acc = vdupq_n_f32(0.0);

    // 32 q4 values are packed into 16 bytes => process two 8-byte chunks.
    for chunk in 0..2usize {
        let packed = vld1_u8(packed_ptr.add(chunk * 8));
        let lo_u = vand_u8(packed, nib_mask);
        let hi_u = vshr_n_u8(packed, 4);
        // Map unsigned nibble [0, 15] -> signed q4 [-8, 7].
        let lo_s = vsub_s8(vreinterpret_s8_u8(veor_u8(lo_u, sign_xor)), sign_sub);
        let hi_s = vsub_s8(vreinterpret_s8_u8(veor_u8(hi_u, sign_xor)), sign_sub);
        let (lo_f0, lo_f1) = s8x8_to_f32x4x2(lo_s);
        let (hi_f0, hi_f1) = s8x8_to_f32x4x2(hi_s);

        let base = chunk * 16;
        let x_pairs0 = vld2q_f32(q_ptr.add(base));
        let x_pairs1 = vld2q_f32(q_ptr.add(base + 8));
        acc = vfmaq_f32(acc, x_pairs0.0, lo_f0);
        acc = vfmaq_f32(acc, x_pairs0.1, hi_f0);
        acc = vfmaq_f32(acc, x_pairs1.0, lo_f1);
        acc = vfmaq_f32(acc, x_pairs1.1, hi_f1);
    }
    vaddvq_f32(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn axpy_q4_block_neon(dst_block: &mut [f32], a: f32, cache: &[u8], elem_offset: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(dst_block.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(elem_offset & 1, 0);

    let packed_ptr = cache.as_ptr().add(elem_offset / 2);
    let dst_ptr = dst_block.as_mut_ptr();
    let nib_mask = vdup_n_u8(0x0f);
    let sign_xor = vdup_n_u8(0x08);
    let sign_sub = vdup_n_s8(8);
    let coeff = vdupq_n_f32(a);

    for chunk in 0..2usize {
        let packed = vld1_u8(packed_ptr.add(chunk * 8));
        let lo_u = vand_u8(packed, nib_mask);
        let hi_u = vshr_n_u8(packed, 4);
        let lo_s = vsub_s8(vreinterpret_s8_u8(veor_u8(lo_u, sign_xor)), sign_sub);
        let hi_s = vsub_s8(vreinterpret_s8_u8(veor_u8(hi_u, sign_xor)), sign_sub);
        let (lo_f0, lo_f1) = s8x8_to_f32x4x2(lo_s);
        let (hi_f0, hi_f1) = s8x8_to_f32x4x2(hi_s);

        let base = chunk * 16;
        let mut dst_pairs0 = vld2q_f32(dst_ptr.add(base));
        let mut dst_pairs1 = vld2q_f32(dst_ptr.add(base + 8));
        dst_pairs0.0 = vfmaq_f32(dst_pairs0.0, coeff, lo_f0);
        dst_pairs0.1 = vfmaq_f32(dst_pairs0.1, coeff, hi_f0);
        dst_pairs1.0 = vfmaq_f32(dst_pairs1.0, coeff, lo_f1);
        dst_pairs1.1 = vfmaq_f32(dst_pairs1.1, coeff, hi_f1);
        vst2q_f32(dst_ptr.add(base), dst_pairs0);
        vst2q_f32(dst_ptr.add(base + 8), dst_pairs1);
    }
}

#[inline]
fn dot_q8_row(q: &[f32], cache: &[i8], row_offset: usize, scale: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_q8_row_neon(q, cache, row_offset, scale);
    }
    #[allow(unreachable_code)]
    {
        let mut acc = 0.0f32;
        for (i, &qv) in q.iter().enumerate() {
            acc += qv * (cache[row_offset + i] as f32 * scale);
        }
        acc
    }
}

#[inline]
fn axpy_q8_row(dst: &mut [f32], a: f32, cache: &[i8], row_offset: usize, scale: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        axpy_q8_row_neon(dst, a, cache, row_offset, scale);
        return;
    }
    #[allow(unreachable_code)]
    {
        let scaled = a * scale;
        for (i, d) in dst.iter_mut().enumerate() {
            *d += scaled * cache[row_offset + i] as f32;
        }
    }
}

#[inline]
unsafe fn dot_q8_row_blocks_ptr(
    q: &[f32],
    cache: &[i8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let n_full = n_blocks.min(q.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        acc += dot_q8_row(&q[b0..b1], cache, row_offset + b0, s0);
        acc += dot_q8_row(&q[b1..b2], cache, row_offset + b1, s1);
        acc += dot_q8_row(&q[b2..b3], cache, row_offset + b2, s2);
        acc += dot_q8_row(&q[b3..b3 + Q4_BLOCK_SIZE], cache, row_offset + b3, s3);
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        acc += dot_q8_row(
            &q[start..end],
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= q.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(q.len());
        acc += dot_q8_row(
            &q[start..end],
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    acc
}

#[inline]
unsafe fn axpy_q8_row_blocks_ptr(
    dst: &mut [f32],
    a: f32,
    cache: &[i8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) {
    let n_full = n_blocks.min(dst.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        axpy_q8_row(&mut dst[b0..b1], a, cache, row_offset + b0, s0);
        axpy_q8_row(&mut dst[b1..b2], a, cache, row_offset + b1, s1);
        axpy_q8_row(&mut dst[b2..b3], a, cache, row_offset + b2, s2);
        axpy_q8_row(
            &mut dst[b3..b3 + Q4_BLOCK_SIZE],
            a,
            cache,
            row_offset + b3,
            s3,
        );
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        axpy_q8_row(
            &mut dst[start..end],
            a,
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= dst.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(dst.len());
        axpy_q8_row(
            &mut dst[start..end],
            a,
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
}

#[inline(always)]
fn dot_q4_full_block(q_block: &[f32], cache: &[u8], elem_offset: usize, scale: f32) -> f32 {
    debug_assert_eq!(q_block.len(), Q4_BLOCK_SIZE);
    #[cfg(target_arch = "aarch64")]
    if (elem_offset & 1) == 0 {
        unsafe {
            return dot_q4_block_neon(q_block, cache, elem_offset) * scale;
        }
    }
    let mut acc = 0.0f32;
    for (i, &qv) in q_block.iter().enumerate() {
        acc += qv * dequant_q4_at(cache, elem_offset + i) as f32 * scale;
    }
    acc
}

#[inline(always)]
fn axpy_q4_full_block(dst_block: &mut [f32], a: f32, cache: &[u8], elem_offset: usize, scale: f32) {
    debug_assert_eq!(dst_block.len(), Q4_BLOCK_SIZE);
    let coeff = a * scale;
    #[cfg(target_arch = "aarch64")]
    if (elem_offset & 1) == 0 {
        unsafe {
            axpy_q4_block_neon(dst_block, coeff, cache, elem_offset);
            return;
        }
    }
    for (i, d) in dst_block.iter_mut().enumerate() {
        *d += coeff * dequant_q4_at(cache, elem_offset + i) as f32;
    }
}

#[inline]
unsafe fn dot_q4_row_ptr(
    q: &[f32],
    cache: &[u8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let n_full = n_blocks.min(q.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        acc += dot_q4_full_block(&q[b0..b1], cache, row_offset + b0, s0);
        acc += dot_q4_full_block(&q[b1..b2], cache, row_offset + b1, s1);
        acc += dot_q4_full_block(&q[b2..b3], cache, row_offset + b2, s2);
        acc += dot_q4_full_block(&q[b3..b3 + Q4_BLOCK_SIZE], cache, row_offset + b3, s3);
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        acc += dot_q4_full_block(
            &q[start..end],
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= q.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(q.len());
        let scale = *scales_ptr.add(bi);
        for (i, &qv) in q[start..end].iter().enumerate() {
            acc += qv * dequant_q4_at(cache, row_offset + start + i) as f32 * scale;
        }
        bi += 1;
    }
    acc
}

#[inline]
unsafe fn axpy_q4_row_ptr(
    dst: &mut [f32],
    a: f32,
    cache: &[u8],
    row_offset: usize,
    scales_ptr: *const f32,
    n_blocks: usize,
) {
    let n_full = n_blocks.min(dst.len() / Q4_BLOCK_SIZE);
    let mut bi = 0usize;

    while bi + 4 <= n_full {
        let s0 = *scales_ptr.add(bi);
        let s1 = *scales_ptr.add(bi + 1);
        let s2 = *scales_ptr.add(bi + 2);
        let s3 = *scales_ptr.add(bi + 3);
        let b0 = bi * Q4_BLOCK_SIZE;
        let b1 = b0 + Q4_BLOCK_SIZE;
        let b2 = b1 + Q4_BLOCK_SIZE;
        let b3 = b2 + Q4_BLOCK_SIZE;
        axpy_q4_full_block(&mut dst[b0..b1], a, cache, row_offset + b0, s0);
        axpy_q4_full_block(&mut dst[b1..b2], a, cache, row_offset + b1, s1);
        axpy_q4_full_block(&mut dst[b2..b3], a, cache, row_offset + b2, s2);
        axpy_q4_full_block(
            &mut dst[b3..b3 + Q4_BLOCK_SIZE],
            a,
            cache,
            row_offset + b3,
            s3,
        );
        bi += 4;
    }
    while bi < n_full {
        let start = bi * Q4_BLOCK_SIZE;
        let end = start + Q4_BLOCK_SIZE;
        axpy_q4_full_block(
            &mut dst[start..end],
            a,
            cache,
            row_offset + start,
            *scales_ptr.add(bi),
        );
        bi += 1;
    }
    while bi < n_blocks {
        let start = bi * Q4_BLOCK_SIZE;
        if start >= dst.len() {
            break;
        }
        let end = (start + Q4_BLOCK_SIZE).min(dst.len());
        let coeff = a * *scales_ptr.add(bi);
        for (i, d) in dst[start..end].iter_mut().enumerate() {
            *d += coeff * dequant_q4_at(cache, row_offset + start + i) as f32;
        }
        bi += 1;
    }
}

#[inline]
fn qwen35_uses_mrope(p: &Config) -> bool {
    p.is_qwen35 && p.rope_sections[0] > 0 && p.rope_sections[1] > 0
}

fn rebuild_rope_cache(p: &Config, s: &mut RunState, pos: usize, is_swa_layer: bool) {
    let current_is_swa = if is_swa_layer { 1 } else { 0 };
    if s.rope_cache_pos == pos as isize && s.rope_cache_is_swa == current_is_swa {
        return;
    }

    let rope_freqs = if p.is_gemma3 && is_swa_layer {
        &s.rope_freqs_swa
    } else {
        &s.rope_freqs
    };
    let rope_half = s.rope_cos.len();

    if qwen35_uses_mrope(p) {
        // llama.cpp M-RoPE text path expands scalar position into [t,h,w,e] = [pos,pos,pos,0].
        let pos_streams = [pos as f32, pos as f32, pos as f32, 0.0f32];
        let section_total = p.rope_sections.iter().sum::<usize>();
        let section_h = p.rope_sections[0];
        let section_w = section_h + p.rope_sections[1];
        let section_e = section_w + p.rope_sections[2];

        for (i, ((cos, sin), &freq)) in s
            .rope_cos
            .iter_mut()
            .zip(s.rope_sin.iter_mut())
            .zip(rope_freqs.iter())
            .take(rope_half)
            .enumerate()
        {
            let sector = i % section_total;
            let pos_value = if sector < section_h {
                pos_streams[0]
            } else if sector < section_w {
                pos_streams[1]
            } else if sector < section_e {
                pos_streams[2]
            } else {
                pos_streams[3]
            };
            let val = pos_value * freq;
            *cos = val.cos();
            *sin = val.sin();
        }
    } else {
        for ((cos, sin), &freq) in s
            .rope_cos
            .iter_mut()
            .zip(s.rope_sin.iter_mut())
            .zip(rope_freqs.iter())
            .take(rope_half)
        {
            let val = pos as f32 * freq;
            *cos = val.cos();
            *sin = val.sin();
        }
    }

    s.rope_cache_pos = pos as isize;
    s.rope_cache_is_swa = current_is_swa;
}

pub(crate) fn malloc_run_state(p: &Config) -> Result<RunState, String> {
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

    let att_len = p
        .n_heads
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing attention buffer size".to_string())?;
    let kv_cache_rows = p
        .n_layers
        .checked_mul(p.seq_len)
        .ok_or_else(|| "overflow while computing kv cache rows".to_string())?;
    let kv_cache_len = kv_cache_rows
        .checked_mul(kv_dim)
        .ok_or_else(|| "overflow while computing kv cache size".to_string())?;
    let kv_cache_q4_len = kv_cache_len.div_ceil(2);

    let requested_mode = kv_cache_mode();
    let (kv_cache_format, key_cache_q8, value_cache_q8, key_cache_q4, value_cache_q4) =
        match requested_mode {
            SwitchKvCacheMode::Q8 => {
                let key = alloc_i8(kv_cache_len, "Q8 key cache")?;
                let value = alloc_i8(kv_cache_len, "Q8 value cache")?;
                (KvCacheFormat::Q8, key, value, Vec::new(), Vec::new())
            }
            SwitchKvCacheMode::Q4 => {
                let key = alloc_u8(kv_cache_q4_len, "Q4 key cache")?;
                let value = alloc_u8(kv_cache_q4_len, "Q4 value cache")?;
                (KvCacheFormat::Q4, Vec::new(), Vec::new(), key, value)
            }
            SwitchKvCacheMode::Auto => {
                let q8_try = (|| -> Result<(Vec<i8>, Vec<i8>), String> {
                    let key = alloc_i8(kv_cache_len, "Q8 key cache")?;
                    let value = alloc_i8(kv_cache_len, "Q8 value cache")?;
                    Ok((key, value))
                })();
                match q8_try {
                    Ok((key, value)) => (KvCacheFormat::Q8, key, value, Vec::new(), Vec::new()),
                    Err(q8_err) => {
                        eprintln!("KV cache Q8 allocation failed: {q8_err}");
                        eprintln!("Falling back to KV cache Q4 format.");
                        let key = alloc_u8(kv_cache_q4_len, "Q4 key cache")?;
                        let value = alloc_u8(kv_cache_q4_len, "Q4 value cache")?;
                        (KvCacheFormat::Q4, Vec::new(), Vec::new(), key, value)
                    }
                }
            }
        };

    Ok(RunState {
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
        att: alloc_f32(att_len, "attention buffer")?,
        logits: vec![0.0; p.vocab_size],
        kv_cache_format,
        key_cache_q8,
        value_cache_q8,
        key_cache_q4,
        value_cache_q4,
        // Q4 mode uses one scale per Q4_BLOCK_SIZE-element block per row. Q8 uses one scale
        // per row. Allocate the larger Q4 layout; Q8 uses only the first kv_cache_rows entries.
        key_cache_scale: alloc_f32(
            kv_cache_rows * (kv_dim / Q4_BLOCK_SIZE).max(1),
            "KV key scale buffer",
        )?,
        value_cache_scale: alloc_f32(
            kv_cache_rows * (kv_dim / Q4_BLOCK_SIZE).max(1),
            "KV value scale buffer",
        )?,
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
        attn_scale: 1.0 / (head_size as f32).sqrt(),
        embed_scale: (p.dim as f32).sqrt(),
    })
}

pub(crate) fn transformer(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(Some(token), None, pos, p, s, w, mapped)
}

pub(crate) fn transformer_with_embedding(
    embedding: &[f32],
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
    mapped: &[u8],
) -> Result<(), String> {
    transformer_inner(None, Some(embedding), pos, p, s, w, mapped)
}

fn transformer_inner(
    token: Option<usize>,
    embedding: Option<&[f32]>,
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
    let mut deepstack_embedding: Option<&[f32]> = None;

    if let Some(input_embedding) = embedding {
        if input_embedding.len() == dim {
            s.x[..dim].copy_from_slice(input_embedding);
        } else if p.n_deepstack_layers > 0 && input_embedding.len() == p.input_embedding_dim {
            s.x[..dim].copy_from_slice(&input_embedding[..dim]);
            deepstack_embedding = Some(&input_embedding[dim..]);
        } else {
            return Err(format!(
                "embedding input length mismatch: got {}, expected {} or {}",
                input_embedding.len(),
                dim,
                p.input_embedding_dim
            ));
        }
    } else {
        let token = token.ok_or_else(|| "missing token input for transformer step".to_string())?;
        let emb_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
        s.x[..dim].copy_from_slice(emb_row);

        if p.is_gemma3 {
            scale_slice_inplace(&mut s.x[..dim], s.embed_scale);
        }
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

            validate_bf16_projection_rows(
                p,
                pos,
                l,
                "attn_k",
                &w.wk[l],
                &s.xb[..dim],
                &s.k[..kv_dim],
                mapped,
            );
            validate_bf16_projection_rows(
                p,
                pos,
                l,
                "attn_v",
                &w.wv[l],
                &s.xb[..dim],
                &s.v[..kv_dim],
                mapped,
            );

            if p.is_qwen2 && !w.attn_q_bias.is_empty() {
                let qb = &w.attn_q_bias[l * q_dim..(l + 1) * q_dim];
                let kb = &w.attn_k_bias[l * kv_dim..(l + 1) * kv_dim];
                let vb = &w.attn_v_bias[l * kv_dim..(l + 1) * kv_dim];
                for (q, &b) in s.q[..q_dim].iter_mut().zip(qb.iter()) {
                    *q += b;
                }
                for ((k, v), (&kbv, &vbv)) in s.k[..kv_dim]
                    .iter_mut()
                    .zip(s.v[..kv_dim].iter_mut())
                    .zip(kb.iter().zip(vb.iter()))
                {
                    *k += kbv;
                    *v += vbv;
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
            let rope_half = s.rope_cos.len();
            rebuild_rope_cache(p, s, pos, is_swa_layer);

            if p.is_gemma3 || p.is_qwen2 || p.is_qwen3vl || p.is_qwen3moe || p.is_qwen3next {
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

            let layer_row_base = l * p.seq_len;
            let row_index = layer_row_base + pos;
            let row_elem_offset = row_index * kv_dim;
            let q8_block_scales = env_flag("GGUF_Q8_BLOCK_SCALES");
            let n_blocks_per_row = kv_dim / Q4_BLOCK_SIZE;

            match s.kv_cache_format {
                KvCacheFormat::Q8 => {
                    if q8_block_scales {
                        let scale_base = row_index * n_blocks_per_row;
                        for b in 0..n_blocks_per_row {
                            let src_start = b * Q4_BLOCK_SIZE;
                            let src_end = src_start + Q4_BLOCK_SIZE;
                            let dst_start = row_elem_offset + src_start;
                            let dst_end = dst_start + Q4_BLOCK_SIZE;
                            quantize_row_q8(
                                &s.k[src_start..src_end],
                                &mut s.key_cache_q8[dst_start..dst_end],
                                &mut s.key_cache_scale[scale_base + b],
                            );
                            quantize_row_q8(
                                &s.v[src_start..src_end],
                                &mut s.value_cache_q8[dst_start..dst_end],
                                &mut s.value_cache_scale[scale_base + b],
                            );
                        }
                    } else {
                        quantize_row_q8(
                            &s.k[..kv_dim],
                            &mut s.key_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                            &mut s.key_cache_scale[row_index],
                        );
                        quantize_row_q8(
                            &s.v[..kv_dim],
                            &mut s.value_cache_q8[row_elem_offset..row_elem_offset + kv_dim],
                            &mut s.value_cache_scale[row_index],
                        );
                    }
                }
                KvCacheFormat::Q4 => {
                    // Quantize in Q4_BLOCK_SIZE-element blocks so each block has its own scale,
                    // preventing outlier activations in one block from zeroing out other blocks.
                    let scale_base = row_index * n_blocks_per_row;
                    for b in 0..n_blocks_per_row {
                        let src_start = b * Q4_BLOCK_SIZE;
                        let elem_off = row_elem_offset + src_start;
                        quantize_q4_block(
                            &s.k[src_start..src_start + Q4_BLOCK_SIZE],
                            &mut s.key_cache_q4,
                            elem_off,
                            &mut s.key_cache_scale[scale_base + b],
                        );
                        quantize_q4_block(
                            &s.v[src_start..src_start + Q4_BLOCK_SIZE],
                            &mut s.value_cache_q4,
                            elem_off,
                            &mut s.value_cache_scale[scale_base + b],
                        );
                    }
                }
            }

            let attn_scale_score = s.attn_scale;
            let apply_attn_scale = !p.is_gemma3;
            let q_all = &s.q[..q_dim];
            let kv_format = s.kv_cache_format;
            let key_cache_q8 = &s.key_cache_q8;
            let value_cache_q8 = &s.value_cache_q8;
            let key_cache_q4 = &s.key_cache_q4;
            let value_cache_q4 = &s.value_cache_q4;
            let key_scales = &s.key_cache_scale;
            let value_scales = &s.value_cache_scale;
            // Number of Q4 scale blocks per full kv_dim row; head_size/Q4_BLOCK_SIZE per KV head.
            let blocks_per_head = head_size / Q4_BLOCK_SIZE;
            let (att_all, xb_all) = (&mut s.att[..p.n_heads * p.seq_len], &mut s.xb[..q_dim]);
            let fuse_qwen35_online_attn = p.online_attn_fusion;

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
                        // Scale slice for this KV head: blocks_per_head blocks starting at
                        // the head's block offset within the row.
                        let head_block_off = kv_head * blocks_per_head;
                        let key_head_scales_ptr =
                            unsafe { key_scales.as_ptr().add(head_block_off) };
                        let value_head_scales_ptr =
                            unsafe { value_scales.as_ptr().add(head_block_off) };
                        if fuse_qwen35_online_attn {
                            xb_head.fill(0.0);
                            let mut max_score = f32::NEG_INFINITY;
                            let mut score_sum = 0.0f32;
                            for t in 0..=pos {
                                let t_row = layer_row_base + t;
                                let row_offset = t_row * kv_dim + kv_head_offset;
                                let mut score = match kv_format {
                                    KvCacheFormat::Q8 => {
                                        if q8_block_scales {
                                            let sb = t_row * n_blocks_per_row;
                                            unsafe {
                                                dot_q8_row_blocks_ptr(
                                                    q_head,
                                                    key_cache_q8,
                                                    row_offset,
                                                    key_head_scales_ptr.add(sb),
                                                    blocks_per_head,
                                                )
                                            }
                                        } else {
                                            dot_q8_row(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_scales[t_row],
                                            )
                                        }
                                    }
                                    KvCacheFormat::Q4 => {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q4_row_ptr(
                                                q_head,
                                                key_cache_q4,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    }
                                };
                                if apply_attn_scale {
                                    score *= attn_scale_score;
                                }

                                if score > max_score {
                                    if score_sum > 0.0 {
                                        let rescale = (max_score - score).exp();
                                        scale_slice_inplace(xb_head, rescale);
                                        score_sum *= rescale;
                                    }
                                    max_score = score;
                                }

                                let weight = (score - max_score).exp();
                                score_sum += weight;
                                match kv_format {
                                    KvCacheFormat::Q8 => {
                                        if q8_block_scales {
                                            let sb = t_row * n_blocks_per_row;
                                            unsafe {
                                                axpy_q8_row_blocks_ptr(
                                                    xb_head,
                                                    weight,
                                                    value_cache_q8,
                                                    row_offset,
                                                    value_head_scales_ptr.add(sb),
                                                    blocks_per_head,
                                                );
                                            }
                                        } else {
                                            axpy_q8_row(
                                                xb_head,
                                                weight,
                                                value_cache_q8,
                                                row_offset,
                                                value_scales[t_row],
                                            );
                                        }
                                    }
                                    KvCacheFormat::Q4 => {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q4_row_ptr(
                                                xb_head,
                                                weight,
                                                value_cache_q4,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    }
                                }
                            }
                            if score_sum > 0.0 {
                                scale_slice_inplace(xb_head, 1.0 / score_sum);
                            }
                        } else {
                            for (t, slot) in att_head.iter_mut().enumerate() {
                                let t_row = layer_row_base + t;
                                let row_offset = t_row * kv_dim + kv_head_offset;
                                let mut score = match kv_format {
                                    KvCacheFormat::Q8 => {
                                        if q8_block_scales {
                                            let sb = t_row * n_blocks_per_row;
                                            unsafe {
                                                dot_q8_row_blocks_ptr(
                                                    q_head,
                                                    key_cache_q8,
                                                    row_offset,
                                                    key_head_scales_ptr.add(sb),
                                                    blocks_per_head,
                                                )
                                            }
                                        } else {
                                            dot_q8_row(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_scales[t_row],
                                            )
                                        }
                                    }
                                    KvCacheFormat::Q4 => {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q4_row_ptr(
                                                q_head,
                                                key_cache_q4,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    }
                                };
                                if apply_attn_scale {
                                    score *= attn_scale_score;
                                }
                                *slot = score;
                            }

                            softmax(att_head, pos + 1);

                            xb_head.fill(0.0);
                            for (t, &a) in att_head.iter().enumerate() {
                                let t_row = layer_row_base + t;
                                let row_offset = t_row * kv_dim + kv_head_offset;
                                match kv_format {
                                    KvCacheFormat::Q8 => {
                                        if q8_block_scales {
                                            let sb = t_row * n_blocks_per_row;
                                            unsafe {
                                                axpy_q8_row_blocks_ptr(
                                                    xb_head,
                                                    a,
                                                    value_cache_q8,
                                                    row_offset,
                                                    value_head_scales_ptr.add(sb),
                                                    blocks_per_head,
                                                );
                                            }
                                        } else {
                                            axpy_q8_row(
                                                xb_head,
                                                a,
                                                value_cache_q8,
                                                row_offset,
                                                value_scales[t_row],
                                            );
                                        }
                                    }
                                    KvCacheFormat::Q4 => {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q4_row_ptr(
                                                xb_head,
                                                a,
                                                value_cache_q4,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    });
            } else {
                for h in 0..p.n_heads {
                    let hs = h * head_size;
                    let q_head = &q_all[hs..hs + head_size];
                    let kv_head = h / kv_mul;
                    let kv_head_offset = kv_head * head_size;
                    let head_block_off = kv_head * blocks_per_head;
                    let key_head_scales_ptr = unsafe { key_scales.as_ptr().add(head_block_off) };
                    let value_head_scales_ptr =
                        unsafe { value_scales.as_ptr().add(head_block_off) };
                    let att_head_full = &mut att_all[h * p.seq_len..(h + 1) * p.seq_len];
                    let att_head = &mut att_head_full[..=pos];
                    let xb_head = &mut xb_all[hs..hs + head_size];
                    if fuse_qwen35_online_attn {
                        xb_head.fill(0.0);
                        let mut max_score = f32::NEG_INFINITY;
                        let mut score_sum = 0.0f32;
                        for t in 0..=pos {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            let mut score = match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q8_row_blocks_ptr(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    } else {
                                        dot_q8_row(
                                            q_head,
                                            key_cache_q8,
                                            row_offset,
                                            key_scales[t_row],
                                        )
                                    }
                                }
                                KvCacheFormat::Q4 => {
                                    let sb = t_row * n_blocks_per_row;
                                    unsafe {
                                        dot_q4_row_ptr(
                                            q_head,
                                            key_cache_q4,
                                            row_offset,
                                            key_head_scales_ptr.add(sb),
                                            blocks_per_head,
                                        )
                                    }
                                }
                            };
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }

                            if score > max_score {
                                if score_sum > 0.0 {
                                    let rescale = (max_score - score).exp();
                                    scale_slice_inplace(xb_head, rescale);
                                    score_sum *= rescale;
                                }
                                max_score = score;
                            }

                            let weight = (score - max_score).exp();
                            score_sum += weight;
                            match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q8_row_blocks_ptr(
                                                xb_head,
                                                weight,
                                                value_cache_q8,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    } else {
                                        axpy_q8_row(
                                            xb_head,
                                            weight,
                                            value_cache_q8,
                                            row_offset,
                                            value_scales[t_row],
                                        );
                                    }
                                }
                                KvCacheFormat::Q4 => {
                                    let sb = t_row * n_blocks_per_row;
                                    unsafe {
                                        axpy_q4_row_ptr(
                                            xb_head,
                                            weight,
                                            value_cache_q4,
                                            row_offset,
                                            value_head_scales_ptr.add(sb),
                                            blocks_per_head,
                                        );
                                    }
                                }
                            }
                        }
                        if score_sum > 0.0 {
                            scale_slice_inplace(xb_head, 1.0 / score_sum);
                        }
                    } else {
                        for (t, slot) in att_head.iter_mut().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            let mut score = match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            dot_q8_row_blocks_ptr(
                                                q_head,
                                                key_cache_q8,
                                                row_offset,
                                                key_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            )
                                        }
                                    } else {
                                        dot_q8_row(
                                            q_head,
                                            key_cache_q8,
                                            row_offset,
                                            key_scales[t_row],
                                        )
                                    }
                                }
                                KvCacheFormat::Q4 => {
                                    let sb = t_row * n_blocks_per_row;
                                    unsafe {
                                        dot_q4_row_ptr(
                                            q_head,
                                            key_cache_q4,
                                            row_offset,
                                            key_head_scales_ptr.add(sb),
                                            blocks_per_head,
                                        )
                                    }
                                }
                            };
                            if apply_attn_scale {
                                score *= attn_scale_score;
                            }
                            *slot = score;
                        }

                        softmax(att_head, pos + 1);

                        xb_head.fill(0.0);
                        for (t, &a) in att_head.iter().enumerate() {
                            let t_row = layer_row_base + t;
                            let row_offset = t_row * kv_dim + kv_head_offset;
                            match kv_format {
                                KvCacheFormat::Q8 => {
                                    if q8_block_scales {
                                        let sb = t_row * n_blocks_per_row;
                                        unsafe {
                                            axpy_q8_row_blocks_ptr(
                                                xb_head,
                                                a,
                                                value_cache_q8,
                                                row_offset,
                                                value_head_scales_ptr.add(sb),
                                                blocks_per_head,
                                            );
                                        }
                                    } else {
                                        axpy_q8_row(
                                            xb_head,
                                            a,
                                            value_cache_q8,
                                            row_offset,
                                            value_scales[t_row],
                                        );
                                    }
                                }
                                KvCacheFormat::Q4 => {
                                    let sb = t_row * n_blocks_per_row;
                                    unsafe {
                                        axpy_q4_row_ptr(
                                            xb_head,
                                            a,
                                            value_cache_q4,
                                            row_offset,
                                            value_head_scales_ptr.add(sb),
                                            blocks_per_head,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if qwen3next_packed_q_gate {
                for h in 0..p.n_heads {
                    let src_base = h * 2 * head_size + head_size;
                    let dst_base = h * head_size;
                    sigmoid_mul_inplace(
                        &mut s.xb[dst_base..dst_base + head_size],
                        &s.hb[src_base..src_base + head_size],
                    );
                }
                sanitize_finite_inplace(&mut s.xb[..q_dim]);
            }
            matmul_quantized(&mut s.xb2[..dim], &s.xb[..q_dim], &w.wo[l], mapped)?;
            sanitize_finite_inplace(&mut s.xb2[..dim]);
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

        if p.is_qwen3moe || (p.is_qwen3next && p.n_experts > 0) {
            let moe_prof = prof_start();
            let expert_hidden = p.expert_hidden_dim;
            let disable_routed = p.is_qwen35 && env_flag("GGUF_QWEN35_DISABLE_ROUTED_EXPERTS");
            let disable_shared = p.is_qwen35 && env_flag("GGUF_QWEN35_DISABLE_SHARED_EXPERT");
            let force_serial_routed =
                p.is_qwen3next && env_flag("GGUF_QWEN3NEXT_SERIAL_ROUTED_EXPERTS");
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

            if !disable_routed {
                let mut routed_selected = Vec::with_capacity(n_selected);
                for j in 0..n_selected {
                    let route_weight = s.moe_topk_weights[j];
                    if route_weight != 0.0 {
                        routed_selected.push((s.moe_topk_indices[j], route_weight));
                    }
                }

                if routed_selected.len() >= 2 && !force_serial_routed {
                    let xb2 = &s.xb2[..dim];
                    let gate_exps = &w.moe_gate_exps[l];
                    let up_exps = &w.moe_up_exps[l];
                    let down_exps = &w.moe_down_exps[l];

                    let per_expert = routed_selected
                        .par_iter()
                        .enumerate()
                        .map(
                            |(order_idx, &(expert_idx, route_weight))| -> Result<(usize, Vec<f32>), String> {
                                let mut hb_local = vec![0.0f32; expert_hidden];
                                let mut hb2_local = vec![0.0f32; expert_hidden];
                                let mut moe_tmp_local = vec![0.0f32; dim];
                                let row_start_ffn = expert_idx * expert_hidden;
                                matmul_quantized_rows(
                                    &mut hb_local[..expert_hidden],
                                    xb2,
                                    gate_exps,
                                    row_start_ffn,
                                    expert_hidden,
                                    mapped,
                                )?;
                                matmul_quantized_rows(
                                    &mut hb2_local[..expert_hidden],
                                    xb2,
                                    up_exps,
                                    row_start_ffn,
                                    expert_hidden,
                                    mapped,
                                )?;
                                silu_and_mul_inplace(
                                    &mut hb_local[..expert_hidden],
                                    &hb2_local[..expert_hidden],
                                );

                                let row_start_down = expert_idx * dim;
                                matmul_quantized_rows(
                                    &mut moe_tmp_local[..dim],
                                    &hb_local[..expert_hidden],
                                    down_exps,
                                    row_start_down,
                                    dim,
                                    mapped,
                                )?;
                                for v in &mut moe_tmp_local[..dim] {
                                    *v *= route_weight;
                                }
                                Ok((order_idx, moe_tmp_local))
                            },
                        )
                        .collect::<Vec<_>>();

                    let mut contributions = Vec::with_capacity(per_expert.len());
                    for item in per_expert {
                        contributions.push(item?);
                    }
                    contributions.sort_by_key(|(order_idx, _)| *order_idx);

                    s.xb[..dim].fill(0.0);
                    for (_, contrib) in contributions {
                        crate::engine::kernels::axpy_inplace(
                            &mut s.xb[..dim],
                            1.0,
                            &contrib[..dim],
                        );
                    }
                } else {
                    for &(expert_idx, route_weight) in &routed_selected {
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

                        silu_and_mul_inplace(&mut s.hb[..expert_hidden], &s.hb2[..expert_hidden]);

                        let row_start_down = expert_idx * dim;
                        matmul_quantized_rows(
                            &mut s.moe_tmp[..dim],
                            &s.hb[..expert_hidden],
                            &w.moe_down_exps[l],
                            row_start_down,
                            dim,
                            mapped,
                        )?;
                        crate::engine::kernels::axpy_inplace(
                            &mut s.xb[..dim],
                            route_weight,
                            &s.moe_tmp[..dim],
                        );
                    }
                }
            }

            if p.is_qwen3next && !disable_shared && !w.moe_shared_gate_inp.is_empty() {
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
                silu_and_mul_inplace(&mut s.hb[..shared_hidden], &s.hb2[..shared_hidden]);
                matmul_quantized(
                    &mut s.moe_tmp[..dim],
                    &s.hb[..shared_hidden],
                    &w.w2[l],
                    mapped,
                )?;
                crate::engine::kernels::axpy_inplace(&mut s.xb[..dim], gate, &s.moe_tmp[..dim]);
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
                silu_and_mul_inplace(&mut s.hb[..hidden_dim], &s.hb2[..hidden_dim]);
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

        if let Some(ds) = deepstack_embedding {
            if l < p.n_deepstack_layers {
                let ds_off = l * dim;
                let ds_end = ds_off + dim;
                if ds_end > ds.len() {
                    return Err(format!(
                        "deepstack embedding length mismatch at layer {l}: need {} values, have {}",
                        ds_end,
                        ds.len()
                    ));
                }
                accum(&mut s.x[..dim], &ds[ds_off..ds_end], dim);
            }
        }
    }

    rmsnorm_inplace(&mut s.x[..dim], &w.rms_final_weight[..dim], dim, eps);
    sanitize_finite_inplace(&mut s.x[..dim]);

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
    sanitize_finite_inplace(&mut s.logits[..p.vocab_size]);

    if p.is_gemma3 && p.final_logit_softcapping > 0.0 {
        let cap = p.final_logit_softcapping;
        for i in 0..p.vocab_size {
            s.logits[i] = cap * (s.logits[i] / cap).tanh();
        }
    }

    Ok(())
}
