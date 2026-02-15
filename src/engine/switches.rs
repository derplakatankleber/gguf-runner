use std::sync::atomic::AtomicU8;
use std::sync::OnceLock;

use crate::cli::CliOptions;

#[cfg(target_arch = "x86_64")]
const PAR_MATMUL_MIN_ROWS_DEFAULT: usize = 384;
#[cfg(not(target_arch = "x86_64"))]
const PAR_MATMUL_MIN_ROWS_DEFAULT: usize = 256;
#[cfg(target_arch = "x86_64")]
const PAR_MATMUL_CHUNK_ROWS_DEFAULT: usize = 64;
#[cfg(not(target_arch = "x86_64"))]
const PAR_MATMUL_CHUNK_ROWS_DEFAULT: usize = 32;
const PAR_ATTN_MIN_HEADS_DEFAULT: usize = 8;
const PAR_QWEN3NEXT_MIN_HEADS_DEFAULT: usize = 8;

static PAR_MATMUL_MIN_ROWS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_MATMUL_CHUNK_ROWS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_ATTN_MIN_HEADS_CFG: OnceLock<usize> = OnceLock::new();
static PAR_QWEN3NEXT_MIN_HEADS_CFG: OnceLock<usize> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_DOTPROD_Q8_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
static AARCH64_QK_MR4_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_AVX2_FMA_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_F16C_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static X86_QK_MR4_CFG: OnceLock<bool> = OnceLock::new();
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
pub(crate) static X86_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
static LAYER_DEBUG_CFG: OnceLock<bool> = OnceLock::new();
static LAYER_DEBUG_POS_CFG: OnceLock<Option<usize>> = OnceLock::new();

#[inline]
pub(crate) fn layer_debug_enabled() -> bool {
    *LAYER_DEBUG_CFG.get_or_init(|| false)
}

#[inline]
pub(crate) fn layer_debug_pos() -> Option<usize> {
    *LAYER_DEBUG_POS_CFG.get_or_init(|| None)
}

#[inline]
pub(crate) fn par_matmul_min_rows() -> usize {
    *PAR_MATMUL_MIN_ROWS_CFG.get_or_init(|| PAR_MATMUL_MIN_ROWS_DEFAULT)
}

#[inline]
pub(crate) fn par_matmul_chunk_rows() -> usize {
    *PAR_MATMUL_CHUNK_ROWS_CFG.get_or_init(|| PAR_MATMUL_CHUNK_ROWS_DEFAULT)
}

#[inline]
pub(crate) fn par_attn_min_heads() -> usize {
    *PAR_ATTN_MIN_HEADS_CFG.get_or_init(|| PAR_ATTN_MIN_HEADS_DEFAULT)
}

#[inline]
pub(crate) fn par_qwen3next_min_heads() -> usize {
    *PAR_QWEN3NEXT_MIN_HEADS_CFG.get_or_init(|| PAR_QWEN3NEXT_MIN_HEADS_DEFAULT)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_dotprod_q8() -> bool {
    *AARCH64_DOTPROD_Q8_CFG.get_or_init(|| false)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_aarch64_qk_mr4() -> bool {
    *AARCH64_QK_MR4_CFG.get_or_init(|| true)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_avx2_fma() -> bool {
    *X86_AVX2_FMA_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_f16c() -> bool {
    *X86_F16C_CFG.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn use_x86_qk_mr4() -> bool {
    *X86_QK_MR4_CFG.get_or_init(|| true)
}

pub(crate) fn init_runtime_config_from_cli(cli: &CliOptions) {
    if let Some(v) = cli.par_matmul_min_rows {
        let _ = PAR_MATMUL_MIN_ROWS_CFG.set(v);
    }
    if let Some(v) = cli.par_matmul_chunk_rows {
        let _ = PAR_MATMUL_CHUNK_ROWS_CFG.set(v);
    }
    if let Some(v) = cli.par_attn_min_heads {
        let _ = PAR_ATTN_MIN_HEADS_CFG.set(v);
    }
    if let Some(v) = cli.par_qwen3next_min_heads {
        let _ = PAR_QWEN3NEXT_MIN_HEADS_CFG.set(v);
    }
    if let Some(v) = cli.layer_debug {
        let _ = LAYER_DEBUG_CFG.set(v);
    }
    if let Some(v) = cli.layer_debug_pos {
        let _ = LAYER_DEBUG_POS_CFG.set(Some(v));
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(v) = cli.aarch64_dotprod_q8 {
            let enabled = v && std::arch::is_aarch64_feature_detected!("dotprod");
            let _ = AARCH64_DOTPROD_Q8_CFG.set(enabled);
        }
        if let Some(v) = cli.aarch64_qk_mr4 {
            let _ = AARCH64_QK_MR4_CFG.set(v);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(v) = cli.x86_avx2 {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
            let _ = X86_AVX2_FMA_CFG.set(enabled);
        }
        if let Some(v) = cli.x86_f16c {
            let enabled = v
                && std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c")
                && std::arch::is_x86_feature_detected!("fma");
            let _ = X86_F16C_CFG.set(enabled);
        }
        if let Some(v) = cli.x86_qk_mr4 {
            let _ = X86_QK_MR4_CFG.set(v);
        }
    }
}
