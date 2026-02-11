use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod model;

const GGUF_MAGIC: u32 = 0x4655_4747;

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const QK4_NL: usize = 32;

const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_Q4_0: i32 = 2;
const GGML_TYPE_Q4_1: i32 = 3;
const GGML_TYPE_Q5_0: i32 = 6;
const GGML_TYPE_Q5_1: i32 = 7;
const GGML_TYPE_Q8_0: i32 = 8;
const GGML_TYPE_Q2_K: i32 = 10;
const GGML_TYPE_Q3_K: i32 = 11;
const GGML_TYPE_Q4_K: i32 = 12;
const GGML_TYPE_Q5_K: i32 = 13;
const GGML_TYPE_Q6_K: i32 = 14;
const GGML_TYPE_IQ4_NL: i32 = 20;
const GGML_TYPE_BF16: i32 = 29;

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const LLAMA3_BOS_TOKEN: i32 = 128000;
const LLAMA3_EOS_TOKEN: i32 = 128001;
const LLAMA3_START_HEADER: i32 = 128006;
const LLAMA3_END_HEADER: i32 = 128007;
const LLAMA3_EOT: i32 = 128009;

const GEMMA3_BOS_TOKEN: i32 = 2;
const GEMMA3_START_TURN: i32 = 106;
const GEMMA3_END_TURN: i32 = 107;

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

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static PROF_TRANSFORMER_NS: AtomicU64 = AtomicU64::new(0);
static PROF_MATMUL_NS: AtomicU64 = AtomicU64::new(0);
static PROF_SSM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static PROF_MOE_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FFN_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FORWARD_PASSES: AtomicU64 = AtomicU64::new(0);
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
static AARCH64_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
static AARCH64_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "aarch64")]
static AARCH64_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
static X86_Q4K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
static X86_Q5K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
#[cfg(target_arch = "x86_64")]
static X86_Q6K_MR4_STATUS: AtomicU8 = AtomicU8::new(0);
static LAZY_MODEL_LOADER: OnceLock<Arc<LazyModelLoader>> = OnceLock::new();

#[inline]
fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

#[inline]
fn parse_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|v| {
            let v = v.trim();
            v.eq_ignore_ascii_case("1")
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(default)
}

#[inline]
fn par_matmul_min_rows() -> usize {
    *PAR_MATMUL_MIN_ROWS_CFG.get_or_init(|| {
        parse_env_usize("LLAMA3PURE_PAR_MATMUL_MIN_ROWS", PAR_MATMUL_MIN_ROWS_DEFAULT)
    })
}

#[inline]
fn par_matmul_chunk_rows() -> usize {
    *PAR_MATMUL_CHUNK_ROWS_CFG.get_or_init(|| {
        parse_env_usize(
            "LLAMA3PURE_PAR_MATMUL_CHUNK_ROWS",
            PAR_MATMUL_CHUNK_ROWS_DEFAULT,
        )
    })
}

#[inline]
fn par_attn_min_heads() -> usize {
    *PAR_ATTN_MIN_HEADS_CFG.get_or_init(|| {
        parse_env_usize("LLAMA3PURE_PAR_ATTN_MIN_HEADS", PAR_ATTN_MIN_HEADS_DEFAULT)
    })
}

#[inline]
fn par_qwen3next_min_heads() -> usize {
    *PAR_QWEN3NEXT_MIN_HEADS_CFG.get_or_init(|| {
        parse_env_usize(
            "LLAMA3PURE_PAR_QWEN3NEXT_MIN_HEADS",
            PAR_QWEN3NEXT_MIN_HEADS_DEFAULT,
        )
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn use_aarch64_dotprod_q8() -> bool {
    *AARCH64_DOTPROD_Q8_CFG.get_or_init(|| {
        parse_env_bool("LLAMA3PURE_AARCH64_DOTPROD_Q8", false)
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn use_aarch64_qk_mr4() -> bool {
    *AARCH64_QK_MR4_CFG.get_or_init(|| parse_env_bool("LLAMA3PURE_AARCH64_QK_MR4", true))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn use_x86_avx2_fma() -> bool {
    *X86_AVX2_FMA_CFG.get_or_init(|| {
        parse_env_bool("LLAMA3PURE_X86_AVX2", true)
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn use_x86_f16c() -> bool {
    *X86_F16C_CFG.get_or_init(|| {
        parse_env_bool("LLAMA3PURE_X86_F16C", true)
            && std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn use_x86_qk_mr4() -> bool {
    *X86_QK_MR4_CFG.get_or_init(|| parse_env_bool("LLAMA3PURE_X86_QK_MR4", true))
}

#[inline(always)]
fn profiling_enabled() -> bool {
    PROFILING_ENABLED.load(AtomicOrdering::Relaxed)
}

#[inline(always)]
fn prof_start() -> Option<Instant> {
    if profiling_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline(always)]
fn prof_end(counter: &AtomicU64, start: Option<Instant>) {
    if let Some(t0) = start {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    }
}

fn profiling_reset() {
    PROF_TRANSFORMER_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MATMUL_NS.store(0, AtomicOrdering::Relaxed);
    PROF_SSM_NS.store(0, AtomicOrdering::Relaxed);
    PROF_ATTN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_MOE_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FFN_NS.store(0, AtomicOrdering::Relaxed);
    PROF_FORWARD_PASSES.store(0, AtomicOrdering::Relaxed);
}

#[cfg(unix)]
const PROT_READ: i32 = 0x1;
#[cfg(unix)]
const MAP_SHARED: i32 = 0x0001;

#[cfg(unix)]
extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GgmlType(i32);

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        Self(v as i32)
    }
}

impl Default for GgmlType {
    fn default() -> Self {
        Self(GGML_TYPE_F32)
    }
}

struct MappedFile {
    ptr: *mut u8,
    len: usize,
}

impl MappedFile {
    #[cfg(unix)]
    fn map(file: &File) -> io::Result<Self> {
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }
        let fd = file.as_raw_fd();
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, fd, 0) };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

const LAZY_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const LAZY_BOOTSTRAP_START_BYTES: usize = 8 * 1024 * 1024;
const LAZY_BOOTSTRAP_MAX_BYTES: usize = 512 * 1024 * 1024;
const LAZY_FETCH_RETRIES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LazyChunkState {
    Missing,
    Fetching,
    Ready,
    Failed,
}

struct LazyModelLoader {
    url: String,
    file: File,
    file_len: usize,
    chunk_bytes: usize,
    chunk_count: usize,
    states: Mutex<Vec<LazyChunkState>>,
    cv: Condvar,
    debug_mode: bool,
    ready_chunks: AtomicUsize,
    fetch_attempts: AtomicUsize,
    fetch_waits: AtomicUsize,
    foreground_fetches: AtomicUsize,
    background_fetches: AtomicUsize,
}

impl LazyModelLoader {
    fn new(url: &str, model_path: &str, debug_mode: bool) -> Result<Self, String> {
        let file_len = Self::probe_remote_len(url)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(model_path)
            .map_err(|e| format!("cannot open local cache file {model_path}: {e}"))?;
        file.set_len(file_len as u64)
            .map_err(|e| format!("cannot size cache file {model_path}: {e}"))?;

        let chunk_bytes = LAZY_CHUNK_BYTES;
        let chunk_count = file_len.div_ceil(chunk_bytes);
        let states = vec![LazyChunkState::Missing; chunk_count];

        Ok(Self {
            url: url.to_string(),
            file,
            file_len,
            chunk_bytes,
            chunk_count,
            states: Mutex::new(states),
            cv: Condvar::new(),
            debug_mode,
            ready_chunks: AtomicUsize::new(0),
            fetch_attempts: AtomicUsize::new(0),
            fetch_waits: AtomicUsize::new(0),
            foreground_fetches: AtomicUsize::new(0),
            background_fetches: AtomicUsize::new(0),
        })
    }

    fn probe_remote_len(url: &str) -> Result<usize, String> {
        if let Ok(resp) = ureq::head(url).call() {
            if let Some(v) = resp.header("Content-Length") {
                if let Ok(n) = v.parse::<u64>() {
                    if n > 0 {
                        return Ok(n as usize);
                    }
                }
            }
        }

        let resp = ureq::get(url)
            .set("Range", "bytes=0-0")
            .call()
            .map_err(|e| format!("cannot query remote size from {url}: {e}"))?;
        if let Some(cr) = resp.header("Content-Range") {
            return Self::parse_content_range_total(cr);
        }
        if let Some(v) = resp.header("Content-Length") {
            if let Ok(n) = v.parse::<u64>() {
                if n > 0 {
                    return Ok(n as usize);
                }
            }
        }
        Err(format!(
            "cannot determine remote size for {url} (missing Content-Length/Content-Range)"
        ))
    }

    fn parse_content_range_total(content_range: &str) -> Result<usize, String> {
        let total = content_range
            .rsplit('/')
            .next()
            .ok_or_else(|| format!("invalid Content-Range header: {content_range}"))?;
        total
            .parse::<u64>()
            .map(|v| v as usize)
            .map_err(|e| format!("invalid Content-Range total in {content_range}: {e}"))
    }

    fn chunk_bounds(&self, chunk_idx: usize) -> (usize, usize) {
        let start = chunk_idx * self.chunk_bytes;
        let end = (start + self.chunk_bytes).min(self.file_len);
        (start, end)
    }

    fn fetch_chunk_into_cache(&self, chunk_idx: usize) -> Result<(), String> {
        let (start, end) = self.chunk_bounds(chunk_idx);
        let mut last_err: Option<String> = None;

        for _ in 0..LAZY_FETCH_RETRIES {
            match self.fetch_chunk_once(start, end) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = Some(e),
            }
        }

        Err(last_err.unwrap_or_else(|| "unknown chunk fetch failure".to_string()))
    }

    fn fetch_chunk_once(&self, start: usize, end: usize) -> Result<(), String> {
        let range = format!("bytes={start}-{}", end - 1);
        let resp = ureq::get(&self.url)
            .set("Range", &range)
            .call()
            .map_err(|e| format!("range request failed ({range}): {e}"))?;

        let status = resp.status();
        if status != 206 && !(status == 200 && start == 0 && end == self.file_len) {
            return Err(format!(
                "unexpected HTTP status {} for range {}",
                status, range
            ));
        }

        let mut body = Vec::with_capacity(end - start);
        resp.into_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("failed reading HTTP body for range {range}: {e}"))?;
        if body.len() != end - start {
            return Err(format!(
                "short read for range {range}: got {} bytes, expected {}",
                body.len(),
                end - start
            ));
        }

        write_all_at(&self.file, &body, start as u64)
            .map_err(|e| format!("failed writing cache bytes [{start}..{end}): {e}"))?;
        Ok(())
    }

    fn ensure_chunk_ready(&self, chunk_idx: usize, is_background: bool) -> Result<(), String> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| "lazy loader state lock poisoned".to_string())?;
        loop {
            match states[chunk_idx] {
                LazyChunkState::Ready => return Ok(()),
                LazyChunkState::Missing | LazyChunkState::Failed => {
                    states[chunk_idx] = LazyChunkState::Fetching;
                    self.fetch_attempts.fetch_add(1, AtomicOrdering::Relaxed);
                    if is_background {
                        self.background_fetches
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        self.foreground_fetches
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    break;
                }
                LazyChunkState::Fetching => {
                    self.fetch_waits.fetch_add(1, AtomicOrdering::Relaxed);
                    states = self
                        .cv
                        .wait(states)
                        .map_err(|_| "lazy loader state lock poisoned".to_string())?;
                }
            }
        }
        drop(states);

        let result = self.fetch_chunk_into_cache(chunk_idx);
        let mut states = self
            .states
            .lock()
            .map_err(|_| "lazy loader state lock poisoned".to_string())?;
        states[chunk_idx] = if result.is_ok() {
            self.ready_chunks.fetch_add(1, AtomicOrdering::Relaxed);
            LazyChunkState::Ready
        } else {
            LazyChunkState::Failed
        };
        self.cv.notify_all();
        result
    }

    fn debug_stats_line(&self) -> String {
        let ready = self.ready_chunks.load(AtomicOrdering::Relaxed);
        let pct = if self.chunk_count == 0 {
            100.0
        } else {
            100.0 * ready as f64 / self.chunk_count as f64
        };
        let attempts = self.fetch_attempts.load(AtomicOrdering::Relaxed);
        let waits = self.fetch_waits.load(AtomicOrdering::Relaxed);
        let fg = self.foreground_fetches.load(AtomicOrdering::Relaxed);
        let bg = self.background_fetches.load(AtomicOrdering::Relaxed);
        format!(
            "Lazy model: ready={}/{} ({:.1}%), fetch_attempts={}, waits={}, fg_fetches={}, bg_fetches={}",
            ready, self.chunk_count, pct, attempts, waits, fg, bg
        )
    }

    fn ensure_range(&self, offset: usize, len: usize) -> Result<(), String> {
        if len == 0 || self.chunk_count == 0 {
            return Ok(());
        }
        if offset >= self.file_len {
            return Err(format!(
                "lazy ensure_range offset {} outside file size {}",
                offset, self.file_len
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "lazy ensure_range overflow".to_string())?
            .min(self.file_len);
        let first = offset / self.chunk_bytes;
        let last = (end - 1) / self.chunk_bytes;
        for c in first..=last {
            self.ensure_chunk_ready(c, false)?;
        }
        Ok(())
    }

    fn start_background_download(self: &Arc<Self>) {
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            for chunk_idx in 0..this.chunk_count {
                if let Err(e) = this.ensure_chunk_ready(chunk_idx, true) {
                    if this.debug_mode {
                        eprintln!(
                            "Lazy model background download stopped at chunk {}: {}",
                            chunk_idx, e
                        );
                    }
                    return;
                }
                if this.debug_mode
                    && (chunk_idx + 1 == this.chunk_count || (chunk_idx + 1) % 128 == 0)
                {
                    eprintln!("{}", this.debug_stats_line());
                }
            }
            if this.debug_mode {
                eprintln!("Lazy model background download finished");
                eprintln!("{}", this.debug_stats_line());
            }
        });
    }
}

fn ensure_model_range(offset: usize, len: usize) -> Result<(), String> {
    if let Some(loader) = LAZY_MODEL_LOADER.get() {
        loader.ensure_range(offset, len)?;
    }
    Ok(())
}

fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        while !buf.is_empty() {
            let n = file.write_at(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write model cache",
                ));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        let _ = buf;
        let _ = offset;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "lazy model cache writes require unix platform",
        ))
    }
}

#[derive(Clone, Debug)]
enum GgufValue {
    UInt(u64),
    Int(i64),
    F32(f32),
    F64(f64),
    Bool(()),
    Str(String),
}

#[derive(Clone, Debug)]
struct Gguftensor {
    name: String,
    n_dims: u32,
    ne: [u64; 4],
    ttype: GgmlType,
    offset: u64,
    data_offset: usize,
}

struct GGUFFile {
    version: u32,
    n_tensors: u64,
    n_kv: u64,
    kv: HashMap<String, GgufValue>,
    tensors: Vec<Gguftensor>,
    tensor_lookup: HashMap<String, usize>,
    tensor_data_start: usize,
    vocab_tokens: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_merges: Vec<String>,
    mapped: MappedFile,
    lazy_loader: Option<Arc<LazyModelLoader>>,
}

impl GGUFFile {
    #[inline]
    fn ensure_range(&self, offset: usize, len: usize) -> Result<(), String> {
        if let Some(loader) = &self.lazy_loader {
            loader.ensure_range(offset, len)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Config {
    dim: usize,
    hidden_dim: usize,
    expert_hidden_dim: usize,
    shared_expert_hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_experts: usize,
    n_experts_used: usize,
    moe_n_group: usize,
    moe_topk_group: usize,
    moe_norm_topk_prob: bool,
    moe_routed_scaling_factor: f32,
    vocab_size: usize,
    seq_len: usize,
    rope_theta: f32,
    head_dim: usize,
    rope_dim: usize,
    is_gemma3: bool,
    is_qwen2: bool,
    is_qwen3moe: bool,
    is_qwen3next: bool,
    final_logit_softcapping: f32,
    rms_norm_eps: f32,
    rope_theta_swa: f32,
    swa_pattern: usize,
    ssm_conv_kernel: usize,
    ssm_inner_size: usize,
    ssm_state_size: usize,
    ssm_time_step_rank: usize,
    ssm_group_count: usize,
}

#[derive(Clone, Default)]
struct QuantizedTensor {
    data_offset: usize,
    ttype: GgmlType,
    rows: usize,
    cols: usize,
}

struct TransformerWeights {
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,
    wq: Vec<QuantizedTensor>,
    wk: Vec<QuantizedTensor>,
    wv: Vec<QuantizedTensor>,
    wo: Vec<QuantizedTensor>,
    w1: Vec<QuantizedTensor>,
    w2: Vec<QuantizedTensor>,
    w3: Vec<QuantizedTensor>,
    attn_qkv: Vec<QuantizedTensor>,
    ssm_ba: Vec<QuantizedTensor>,
    ssm_conv1d: Vec<Vec<f32>>,
    ssm_a: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    moe_gate_inp: Vec<QuantizedTensor>,
    moe_gate_exps: Vec<QuantizedTensor>,
    moe_up_exps: Vec<QuantizedTensor>,
    moe_down_exps: Vec<QuantizedTensor>,
    moe_shared_gate_inp: Vec<f32>,
    rms_final_weight: Vec<f32>,
    wcls: QuantizedTensor,
    wcls_is_embed: bool,
    attn_q_bias: Vec<f32>,
    attn_k_bias: Vec<f32>,
    attn_v_bias: Vec<f32>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    attn_qk_norm_present: Vec<bool>,
    attn_post_norm: Vec<f32>,
    ffn_post_norm: Vec<f32>,
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    moe_tmp: Vec<f32>,
    moe_logits: Vec<f32>,
    moe_topk_indices: Vec<usize>,
    moe_topk_weights: Vec<f32>,
    moe_scores: Vec<f32>,
    moe_selected_group: Vec<bool>,
    moe_group_scores: Vec<f32>,
    moe_group_rank: Vec<usize>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ssm_qkv: Vec<f32>,
    ssm_conv: Vec<f32>,
    ssm_q: Vec<f32>,
    ssm_k: Vec<f32>,
    ssm_v: Vec<f32>,
    ssm_z: Vec<f32>,
    ssm_ba: Vec<f32>,
    ssm_gate_exp: Vec<f32>,
    ssm_beta: Vec<f32>,
    ssm_proj: Vec<f32>,
    ssm_kv_mem: Vec<f32>,
    ssm_delta: Vec<f32>,
    ssm_conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    rope_freqs: Vec<f32>,
    rope_freqs_swa: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_cache_pos: isize,
    rope_cache_is_swa: isize,
    head_size: usize,
    kv_dim: usize,
    q_dim: usize,
    kv_mul: usize,
    kv_cache_layer_size: usize,
    attn_scale: f32,
    embed_scale: f32,
}

#[derive(Default)]
struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_size: usize,
    max_token_length: usize,
    bos_token: i32,
    eos_token: i32,
    start_header_token: i32,
    end_header_token: i32,
    eot_token: i32,
    use_sentencepiece: bool,
    token_to_id: HashMap<String, i32>,
    merges: Vec<String>,
    merge_ranks: HashMap<String, usize>,
}

struct XorShiftRng {
    seed: u64,
}

impl XorShiftRng {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn random_u32(&mut self) -> u32 {
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        ((self.seed.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }

    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }
}

#[inline]
fn read_u16_le(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
fn read_f32_le(data: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
fn fp16_to_fp32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1f) as i32;
    let mut mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            exp += 1;
            mant &= !0x0400;
            let exp32 = (exp + (127 - 15)) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let exp32 = (exp + (127 - 15)) as u32;
        sign | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

#[inline]
fn bf16_to_fp32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn read_exact_array<const N: usize>(r: &mut File) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u8(r: &mut File) -> io::Result<u8> {
    Ok(read_exact_array::<1>(r)?[0])
}

fn read_i8(r: &mut File) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut File) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_i16(r: &mut File) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_exact_array::<2>(r)?))
}

fn read_u32(r: &mut File) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_i32(r: &mut File) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_u64(r: &mut File) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_i64(r: &mut File) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_f32(r: &mut File) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_exact_array::<4>(r)?))
}

fn read_f64(r: &mut File) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_exact_array::<8>(r)?))
}

fn read_bool(r: &mut File) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_gguf_string(r: &mut File) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    match String::from_utf8(buf) {
        Ok(s) => Ok(s),
        Err(e) => Ok(String::from_utf8_lossy(e.as_bytes()).into_owned()),
    }
}

fn skip_gguf_value(r: &mut File, value_type: u32) -> io::Result<()> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => {
            let _ = read_u8(r)?;
        }
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
            let _ = read_u16(r)?;
        }
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
            let _ = read_u32(r)?;
        }
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
            let _ = read_u64(r)?;
        }
        GGUF_TYPE_STRING => {
            let _ = read_gguf_string(r)?;
        }
        GGUF_TYPE_ARRAY => {
            let arr_type = read_u32(r)?;
            let arr_len = read_u64(r)?;
            for _ in 0..arr_len {
                skip_gguf_value(r, arr_type)?;
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported gguf value type: {value_type}"),
            ));
        }
    }
    Ok(())
}

fn read_gguf_scalar(r: &mut File, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::UInt(read_u8(r)? as u64)),
        GGUF_TYPE_INT8 => Ok(GgufValue::Int(read_i8(r)? as i64)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::UInt(read_u16(r)? as u64)),
        GGUF_TYPE_INT16 => Ok(GgufValue::Int(read_i16(r)? as i64)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::UInt(read_u32(r)? as u64)),
        GGUF_TYPE_INT32 => Ok(GgufValue::Int(read_i32(r)? as i64)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::UInt(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::Int(read_i64(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_BOOL => {
            let _ = read_bool(r)?;
            Ok(GgufValue::Bool(()))
        }
        GGUF_TYPE_STRING => Ok(GgufValue::Str(read_gguf_string(r)?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported scalar gguf type: {value_type}"),
        )),
    }
}

fn parse_gguf_file_local(filename: &str, debug_mode: bool) -> Result<GGUFFile, String> {
    let mut file = File::open(filename).map_err(|e| format!("cannot open file {filename}: {e}"))?;

    let magic = read_u32(&mut file).map_err(|e| format!("failed to read magic number: {e}"))?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "invalid GGUF magic: expected 0x{GGUF_MAGIC:X}, got 0x{magic:X}"
        ));
    }

    let version = read_u32(&mut file).map_err(|e| format!("failed to read version: {e}"))?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version: {version}"));
    }

    let n_tensors = read_u64(&mut file).map_err(|e| format!("failed to read n_tensors: {e}"))?;
    let n_kv = read_u64(&mut file).map_err(|e| format!("failed to read n_kv: {e}"))?;

    if debug_mode {
        eprintln!("GGUF version: {version}, tensors: {n_tensors}, kv pairs: {n_kv}");
    }

    let mut kv: HashMap<String, GgufValue> = HashMap::new();
    let mut vocab_tokens: Vec<String> = Vec::new();
    let mut vocab_scores: Vec<f32> = Vec::new();
    let mut vocab_merges: Vec<String> = Vec::new();

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut file).map_err(|e| format!("failed to read key: {e}"))?;
        let value_type =
            read_u32(&mut file).map_err(|e| format!("failed to read value type: {e}"))?;

        if value_type == GGUF_TYPE_ARRAY {
            let arr_type =
                read_u32(&mut file).map_err(|e| format!("failed to read array type: {e}"))?;
            let arr_len =
                read_u64(&mut file).map_err(|e| format!("failed to read array len: {e}"))?;

            if key == "tokenizer.ggml.tokens" && arr_type == GGUF_TYPE_STRING {
                vocab_tokens.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let tok = read_gguf_string(&mut file)
                        .map_err(|e| format!("failed to read token: {e}"))?;
                    vocab_tokens.push(tok);
                }
            } else if key == "tokenizer.ggml.scores" && arr_type == GGUF_TYPE_FLOAT32 {
                vocab_scores.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let score =
                        read_f32(&mut file).map_err(|e| format!("failed to read score: {e}"))?;
                    vocab_scores.push(score);
                }
            } else if key == "tokenizer.ggml.merges" && arr_type == GGUF_TYPE_STRING {
                vocab_merges.reserve(arr_len as usize);
                for _ in 0..arr_len {
                    let merge = read_gguf_string(&mut file)
                        .map_err(|e| format!("failed to read merge: {e}"))?;
                    vocab_merges.push(merge);
                }
            } else {
                for _ in 0..arr_len {
                    skip_gguf_value(&mut file, arr_type)
                        .map_err(|e| format!("failed to skip array value for key {key}: {e}"))?;
                }
            }
        } else {
            let value = read_gguf_scalar(&mut file, value_type)
                .map_err(|e| format!("failed to read scalar for key {key}: {e}"))?;
            kv.insert(key, value);
        }
    }

    let mut tensors: Vec<Gguftensor> = Vec::with_capacity(n_tensors as usize);

    for _ in 0..n_tensors {
        let name =
            read_gguf_string(&mut file).map_err(|e| format!("failed to read tensor name: {e}"))?;
        let n_dims = read_u32(&mut file).map_err(|e| format!("failed to read n_dims: {e}"))?;
        if n_dims > 4 {
            return Err(format!("tensor {name} has unsupported n_dims={n_dims}"));
        }

        let mut ne = [1u64; 4];
        for i in 0..n_dims as usize {
            ne[i] = read_u64(&mut file).map_err(|e| format!("failed reading tensor dims: {e}"))?;
        }

        let ttype = GgmlType::from_u32(
            read_u32(&mut file).map_err(|e| format!("failed reading tensor type: {e}"))?,
        );
        let offset =
            read_u64(&mut file).map_err(|e| format!("failed reading tensor offset: {e}"))?;

        tensors.push(Gguftensor {
            name,
            n_dims,
            ne,
            ttype,
            offset,
            data_offset: 0,
        });
    }

    let header_end = file
        .stream_position()
        .map_err(|e| format!("failed to query header end: {e}"))?;

    let alignment = get_gguf_int_from_map(&kv, "general.alignment", 32) as u64;
    let tensor_data_offset = ((header_end + alignment - 1) / alignment) * alignment;

    let mapped = MappedFile::map(&file).map_err(|e| format!("mmap failed: {e}"))?;
    let mapped_len = mapped.len;

    let mut tensor_lookup = HashMap::new();
    for (idx, t) in tensors.iter_mut().enumerate() {
        let abs_off = tensor_data_offset as usize + t.offset as usize;
        if abs_off >= mapped_len {
            return Err(format!("tensor {} points outside mapped file", t.name));
        }
        t.data_offset = abs_off;
        tensor_lookup.insert(t.name.clone(), idx);
    }

    if !vocab_tokens.is_empty() && vocab_scores.is_empty() {
        vocab_scores = vec![0.0; vocab_tokens.len()];
    }

    Ok(GGUFFile {
        version,
        n_tensors,
        n_kv,
        kv,
        tensors,
        tensor_lookup,
        tensor_data_start: tensor_data_offset as usize,
        vocab_tokens,
        vocab_scores,
        vocab_merges,
        mapped,
        lazy_loader: None,
    })
}

fn parse_gguf_file(
    filename: &str,
    model_url: Option<&str>,
    debug_mode: bool,
) -> Result<GGUFFile, String> {
    let model_path = Path::new(filename);
    let local_exists = model_path.exists();

    if local_exists {
        match parse_gguf_file_local(filename, debug_mode) {
            Ok(gguf) => {
                if debug_mode {
                    eprintln!("Using local model file: {filename}");
                }
                return Ok(gguf);
            }
            Err(e) => {
                if model_url.is_none() {
                    return Err(e);
                }
                eprintln!(
                    "Warning: local model file parse failed ({}). Falling back to remote lazy load.",
                    e
                );
            }
        }
    } else if model_url.is_none() {
        return Err(format!(
            "model file not found: {} (provide -url to lazily fetch it)",
            filename
        ));
    }

    let url = model_url.ok_or_else(|| "missing model url".to_string())?;
    let loader = Arc::new(LazyModelLoader::new(url, filename, debug_mode)?);
    let _ = LAZY_MODEL_LOADER.set(Arc::clone(&loader));

    let mut bootstrap = LAZY_BOOTSTRAP_START_BYTES.min(loader.file_len.max(1));
    let max_bootstrap = LAZY_BOOTSTRAP_MAX_BYTES.min(loader.file_len.max(1));

    loop {
        if debug_mode {
            eprintln!(
                "Lazy model bootstrap: ensuring first {} bytes of {}",
                bootstrap, loader.file_len
            );
        }
        loader.ensure_range(0, bootstrap)?;
        match parse_gguf_file_local(filename, debug_mode) {
            Ok(mut gguf) => {
                gguf.lazy_loader = Some(Arc::clone(&loader));
                loader.start_background_download();
                if debug_mode {
                    eprintln!(
                        "Lazy model mode enabled: url={}, local_cache={}, bytes_bootstrapped={}",
                        url, filename, bootstrap
                    );
                }
                return Ok(gguf);
            }
            Err(e) => {
                if bootstrap >= max_bootstrap {
                    return Err(format!(
                        "failed to parse GGUF metadata after bootstrapping {} bytes: {}",
                        bootstrap, e
                    ));
                }
                let next = (bootstrap.saturating_mul(2)).min(max_bootstrap);
                if next == bootstrap {
                    return Err(format!(
                        "failed to parse GGUF metadata at {} bytes: {}",
                        bootstrap, e
                    ));
                }
                if debug_mode {
                    eprintln!(
                        "Lazy model bootstrap parse retry: {} -> {} bytes ({})",
                        bootstrap, next, e
                    );
                }
                bootstrap = next;
            }
        }
    }
}

fn get_gguf_int_from_map(kv: &HashMap<String, GgufValue>, key: &str, default_val: i64) -> i64 {
    match kv.get(key) {
        Some(GgufValue::UInt(v)) => *v as i64,
        Some(GgufValue::Int(v)) => *v,
        _ => default_val,
    }
}

fn get_gguf_float_from_map(kv: &HashMap<String, GgufValue>, key: &str, default_val: f32) -> f32 {
    match kv.get(key) {
        Some(GgufValue::F32(v)) => *v,
        Some(GgufValue::F64(v)) => *v as f32,
        _ => default_val,
    }
}

fn get_gguf_string_from_map<'a>(kv: &'a HashMap<String, GgufValue>, key: &str) -> Option<&'a str> {
    match kv.get(key) {
        Some(GgufValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn find_gguf_tensor<'a>(gguf: &'a GGUFFile, name: &str) -> Option<&'a Gguftensor> {
    gguf.tensor_lookup
        .get(name)
        .and_then(|idx| gguf.tensors.get(*idx))
}

fn get_block_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16 | 30 => 1,
        GGML_TYPE_Q4_0 => QK4_0,
        GGML_TYPE_Q4_1 => QK4_1,
        GGML_TYPE_Q5_0 => QK5_0,
        GGML_TYPE_Q5_1 => QK5_1,
        GGML_TYPE_Q8_0 => QK8_0,
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => QK_K,
        GGML_TYPE_IQ4_NL => QK4_NL,
        _ => 1,
    }
}

fn get_type_size(ttype: GgmlType) -> usize {
    match ttype.0 {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 | GGML_TYPE_BF16 | 30 => 2,
        GGML_TYPE_Q4_0 => 2 + QK4_0 / 2,
        GGML_TYPE_Q4_1 => 2 + 2 + QK4_1 / 2,
        GGML_TYPE_Q5_0 => 2 + 4 + QK5_0 / 2,
        GGML_TYPE_Q5_1 => 2 + 2 + 4 + QK5_1 / 2,
        GGML_TYPE_Q8_0 => 2 + QK8_0,
        GGML_TYPE_Q2_K => QK_K / 16 + QK_K / 4 + 2 + 2,
        GGML_TYPE_Q3_K => QK_K / 8 + QK_K / 4 + 12 + 2,
        GGML_TYPE_Q4_K => 2 + 2 + 12 + QK_K / 2,
        GGML_TYPE_Q5_K => 2 + 2 + 12 + QK_K / 8 + QK_K / 2,
        GGML_TYPE_Q6_K => QK_K / 2 + QK_K / 4 + QK_K / 16 + 2,
        GGML_TYPE_IQ4_NL => 2 + QK4_NL / 2,
        _ => 0,
    }
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequantize_row_q4_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let nb = k / QK4_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qs = &src[off + 2..off + 2 + QK4_0 / 2];
        for j in 0..QK4_0 / 2 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            dst[i * QK4_0 + j] = x0 as f32 * d;
            dst[i * QK4_0 + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
}

fn dequantize_row_q4_1(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let nb = k / QK4_1;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let m = fp16_to_fp32(read_u16_le(src, off + 2));
        let qs = &src[off + 4..off + 4 + QK4_1 / 2];
        for j in 0..QK4_1 / 2 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            dst[i * QK4_1 + j] = x0 * d + m;
            dst[i * QK4_1 + j + QK4_1 / 2] = x1 * d + m;
        }
    }
}

fn dequantize_row_q5_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let nb = k / QK5_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qh = read_u32_le(src, off + 2);
        let qs = &src[off + 6..off + 6 + QK5_0 / 2];
        for j in 0..QK5_0 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
            dst[i * QK5_0 + j] = x0 as f32 * d;
            dst[i * QK5_0 + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
}

fn dequantize_row_q5_1(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let nb = k / QK5_1;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let m = fp16_to_fp32(read_u16_le(src, off + 2));
        let qh = read_u32_le(src, off + 4);
        let qs = &src[off + 8..off + 8 + QK5_1 / 2];
        for j in 0..QK5_1 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as f32;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as f32;
            dst[i * QK5_1 + j] = x0 * d + m;
            dst[i * QK5_1 + j + QK5_1 / 2] = x1 * d + m;
        }
    }
}

fn dequantize_row_q8_0(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let nb = k / QK8_0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        for j in 0..QK8_0 {
            let q = src[off + 2 + j] as i8;
            dst[i * QK8_0 + j] = q as f32 * d;
        }
    }
}

fn dequantize_row_q4_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let nb = k / QK_K;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let dmin = fp16_to_fp32(read_u16_le(src, off + 2));
        let scales = &src[off + 4..off + 16];
        let mut q_off = off + 16;
        let mut y_idx = i * QK_K;
        let mut is = 0usize;
        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &src[q_off..q_off + 32];
            for l in 0..32 {
                dst[y_idx] = d1 * (q[l] & 0x0f) as f32 - m1f;
                y_idx += 1;
            }
            for l in 0..32 {
                dst[y_idx] = d2 * (q[l] >> 4) as f32 - m2f;
                y_idx += 1;
            }
            q_off += 32;
            is += 2;
        }
    }
}

fn dequantize_row_q2_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let nb = k / QK_K;
    for i in 0..nb {
        let off = i * block_sz;
        let scales = &src[off..off + QK_K / 16];
        let mut q_off = off + QK_K / 16;
        let d = fp16_to_fp32(read_u16_le(src, off + QK_K / 16 + QK_K / 4));
        let dmin = fp16_to_fp32(read_u16_le(src, off + QK_K / 16 + QK_K / 4 + 2));

        let mut is = 0usize;
        let mut y_idx = i * QK_K;

        for _ in (0..QK_K).step_by(128) {
            let q = &src[q_off..q_off + 32];
            let mut shift = 0;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let mut dl = d * (sc & 0x0f) as f32;
                let mut ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    dst[y_idx] = dl * ((q[l] >> shift) & 0x03) as f32 - ml;
                    y_idx += 1;
                }

                let sc2 = scales[is];
                is += 1;
                dl = d * (sc2 & 0x0f) as f32;
                ml = dmin * (sc2 >> 4) as f32;
                for l in 0..16 {
                    dst[y_idx] = dl * ((q[l + 16] >> shift) & 0x03) as f32 - ml;
                    y_idx += 1;
                }

                shift += 2;
            }
            q_off += 32;
        }
    }
}

fn q3_scales(scales12: &[u8]) -> [i8; 16] {
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..12 {
        let idx = i / 4;
        aux[idx] |= (scales12[i] as u32) << ((i % 4) * 8);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    let mut out = [0i8; 16];
    for i in 0..4 {
        let b = aux[i].to_le_bytes();
        out[i * 4] = b[0] as i8;
        out[i * 4 + 1] = b[1] as i8;
        out[i * 4 + 2] = b[2] as i8;
        out[i * 4 + 3] = b[3] as i8;
    }
    out
}

fn dequantize_row_q3_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let hmask = &src[off..off + QK_K / 8];
        let mut q_off = off + QK_K / 8;
        let scales = q3_scales(&src[off + QK_K / 8 + QK_K / 4..off + QK_K / 8 + QK_K / 4 + 12]);
        let d_all = fp16_to_fp32(read_u16_le(src, off + QK_K / 8 + QK_K / 4 + 12));

        let mut is = 0usize;
        let mut y_idx = i * QK_K;
        let mut m: u8 = 1;

        for _ in (0..QK_K).step_by(128) {
            let q = &src[q_off..q_off + 32];
            let mut shift = 0usize;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let w = ((q[l] >> shift) & 3) as i8 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                    dst[y_idx] = dl * w as f32;
                    y_idx += 1;
                }

                let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let w = ((q[l + 16] >> shift) & 3) as i8
                        - if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    dst[y_idx] = dl2 * w as f32;
                    y_idx += 1;
                }

                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }
}

fn dequantize_row_q5_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let dmin = fp16_to_fp32(read_u16_le(src, off + 2));
        let scales = &src[off + 4..off + 16];
        let qh = &src[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut y_idx = i * QK_K;

        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            let ql = &src[ql_off..ql_off + 32];

            for l in 0..32 {
                let v = (ql[l] & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                dst[y_idx] = d1 * v as f32 - m1f;
                y_idx += 1;
            }
            for l in 0..32 {
                let v = (ql[l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                dst[y_idx] = d2 * v as f32 - m2f;
                y_idx += 1;
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequantize_row_q6_k(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let nb = k / QK_K;

    for i in 0..nb {
        let off = i * block_sz;
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let d = fp16_to_fp32(read_u16_le(src, off + QK_K / 2 + QK_K / 4 + QK_K / 16));

        let mut y_idx = i * QK_K;
        for _ in (0..QK_K).step_by(128) {
            let ql = &src[ql_off..ql_off + 64];
            let qh = &src[qh_off..qh_off + 32];
            let sc = &src[sc_off..sc_off + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                dst[y_idx + l] = d * sc[is] as i8 as f32 * q1 as f32;
                dst[y_idx + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                dst[y_idx + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                dst[y_idx + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            y_idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

fn dequantize_row_f16(src: &[u8], dst: &mut [f32], k: usize) {
    for i in 0..k {
        dst[i] = fp16_to_fp32(read_u16_le(src, i * 2));
    }
}

fn dequantize_row_bf16(src: &[u8], dst: &mut [f32], k: usize) {
    for i in 0..k {
        dst[i] = bf16_to_fp32(read_u16_le(src, i * 2));
    }
}

fn dequantize_row_iq4_nl(src: &[u8], dst: &mut [f32], k: usize) {
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let nb = k / QK4_NL;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(src, off));
        let qs = &src[off + 2..off + 2 + QK4_NL / 2];
        for j in 0..QK4_NL / 2 {
            dst[i * QK4_NL + j] = d * KVALUES_IQ4NL[(qs[j] & 0x0f) as usize] as f32;
            dst[i * QK4_NL + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

fn dequantize_tensor(src: &[u8], n_elements: usize, ttype: GgmlType) -> Result<Vec<f32>, String> {
    let mut dst = vec![0.0; n_elements];
    match ttype.0 {
        GGML_TYPE_F32 => {
            for i in 0..n_elements {
                dst[i] = read_f32_le(src, i * 4);
            }
        }
        GGML_TYPE_F16 => dequantize_row_f16(src, &mut dst, n_elements),
        GGML_TYPE_Q4_0 => dequantize_row_q4_0(src, &mut dst, n_elements),
        GGML_TYPE_Q4_1 => dequantize_row_q4_1(src, &mut dst, n_elements),
        GGML_TYPE_Q5_0 => dequantize_row_q5_0(src, &mut dst, n_elements),
        GGML_TYPE_Q5_1 => dequantize_row_q5_1(src, &mut dst, n_elements),
        GGML_TYPE_Q8_0 => dequantize_row_q8_0(src, &mut dst, n_elements),
        GGML_TYPE_Q2_K => dequantize_row_q2_k(src, &mut dst, n_elements),
        GGML_TYPE_Q3_K => dequantize_row_q3_k(src, &mut dst, n_elements),
        GGML_TYPE_Q4_K => dequantize_row_q4_k(src, &mut dst, n_elements),
        GGML_TYPE_Q5_K => dequantize_row_q5_k(src, &mut dst, n_elements),
        GGML_TYPE_Q6_K => dequantize_row_q6_k(src, &mut dst, n_elements),
        GGML_TYPE_IQ4_NL => dequantize_row_iq4_nl(src, &mut dst, n_elements),
        GGML_TYPE_BF16 | 30 => dequantize_row_bf16(src, &mut dst, n_elements),
        _ => return Err(format!("unsupported quantization type: {}", ttype.0)),
    }
    Ok(dst)
}

#[inline(always)]
fn dot_f32_scalar_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i < n {
        unsafe {
            sum += *a.add(i) * *b.add(i);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_f32_simd_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i + 16 <= n {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        let a1 = vld1q_f32(a.add(i + 4));
        let b1 = vld1q_f32(b.add(i + 4));
        let a2 = vld1q_f32(a.add(i + 8));
        let b2 = vld1q_f32(b.add(i + 8));
        let a3 = vld1q_f32(a.add(i + 12));
        let b3 = vld1q_f32(b.add(i + 12));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
        i += 16;
    }
    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while i + 4 <= n {
        let av = vld1q_f32(a.add(i));
        let bv = vld1q_f32(b.add(i));
        acc = vfmaq_f32(acc, av, bv);
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm256_loadu_ps(b.add(i));
        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm256_loadu_ps(b.add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().copied().sum::<f32>();
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn vec_dot_f16_f16c_prefix(x: *const f32, w: *const u8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let hv = _mm_loadu_si128(w.add(i * 2) as *const __m128i);
        let wv = _mm256_cvtph_ps(hv);
        acc = _mm256_fmadd_ps(xv, wv, acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().copied().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_bf16_avx2_prefix(x: *const f32, w: *const u8, n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let hv = _mm_loadu_si128(w.add(i * 2) as *const __m128i);
        let wv_i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hv), 16);
        let wv = _mm256_castsi256_ps(wv_i32);
        acc = _mm256_fmadd_ps(xv, wv, acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().copied().sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_f32_simd_ptr(a: *const f32, b: *const f32, n: usize) -> f32 {
    if use_x86_avx2_fma() {
        return dot_f32_avx2_ptr(a, b, n);
    }
    let mut i = 0usize;
    let mut acc = _mm_setzero_ps();
    while i + 4 <= n {
        let av = _mm_loadu_ps(a.add(i));
        let bv = _mm_loadu_ps(b.add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

#[inline(always)]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        return dot_f32_simd_ptr(a.as_ptr(), b.as_ptr(), a.len());
    }
    #[allow(unreachable_code)]
    dot_f32_scalar_ptr(a.as_ptr(), b.as_ptr(), a.len())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn axpy_simd_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    let mut i = 0usize;
    let av = vdupq_n_f32(a);
    while i + 16 <= n {
        let dv0 = vld1q_f32(dst.add(i));
        let sv0 = vld1q_f32(src.add(i));
        let dv1 = vld1q_f32(dst.add(i + 4));
        let sv1 = vld1q_f32(src.add(i + 4));
        let dv2 = vld1q_f32(dst.add(i + 8));
        let sv2 = vld1q_f32(src.add(i + 8));
        let dv3 = vld1q_f32(dst.add(i + 12));
        let sv3 = vld1q_f32(src.add(i + 12));
        vst1q_f32(dst.add(i), vfmaq_f32(dv0, sv0, av));
        vst1q_f32(dst.add(i + 4), vfmaq_f32(dv1, sv1, av));
        vst1q_f32(dst.add(i + 8), vfmaq_f32(dv2, sv2, av));
        vst1q_f32(dst.add(i + 12), vfmaq_f32(dv3, sv3, av));
        i += 16;
    }
    while i + 4 <= n {
        let dv = vld1q_f32(dst.add(i));
        let sv = vld1q_f32(src.add(i));
        let out = vfmaq_f32(dv, sv, av);
        vst1q_f32(dst.add(i), out);
        i += 4;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    let mut i = 0usize;
    let av = _mm256_set1_ps(a);
    while i + 16 <= n {
        let dv0 = _mm256_loadu_ps(dst.add(i));
        let sv0 = _mm256_loadu_ps(src.add(i));
        let dv1 = _mm256_loadu_ps(dst.add(i + 8));
        let sv1 = _mm256_loadu_ps(src.add(i + 8));
        _mm256_storeu_ps(dst.add(i), _mm256_fmadd_ps(sv0, av, dv0));
        _mm256_storeu_ps(dst.add(i + 8), _mm256_fmadd_ps(sv1, av, dv1));
        i += 16;
    }
    while i + 8 <= n {
        let dv = _mm256_loadu_ps(dst.add(i));
        let sv = _mm256_loadu_ps(src.add(i));
        _mm256_storeu_ps(dst.add(i), _mm256_fmadd_ps(sv, av, dv));
        i += 8;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn axpy_simd_ptr(dst: *mut f32, src: *const f32, a: f32, n: usize) {
    if use_x86_avx2_fma() {
        return axpy_avx2_ptr(dst, src, a, n);
    }
    let mut i = 0usize;
    let av = _mm_set1_ps(a);
    while i + 4 <= n {
        let dv = _mm_loadu_ps(dst.add(i));
        let sv = _mm_loadu_ps(src.add(i));
        let out = _mm_add_ps(dv, _mm_mul_ps(sv, av));
        _mm_storeu_ps(dst.add(i), out);
        i += 4;
    }
    while i < n {
        *dst.add(i) += a * *src.add(i);
        i += 1;
    }
}

#[inline(always)]
fn axpy_inplace(dst: &mut [f32], a: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        axpy_simd_ptr(dst.as_mut_ptr(), src.as_ptr(), a, dst.len());
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..dst.len() {
        dst[i] += a * src[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scale_simd_inplace(x: *mut f32, alpha: f32, n: usize) {
    let mut i = 0usize;
    let av = vdupq_n_f32(alpha);
    while i + 16 <= n {
        let xv0 = vld1q_f32(x.add(i));
        let xv1 = vld1q_f32(x.add(i + 4));
        let xv2 = vld1q_f32(x.add(i + 8));
        let xv3 = vld1q_f32(x.add(i + 12));
        vst1q_f32(x.add(i), vmulq_f32(xv0, av));
        vst1q_f32(x.add(i + 4), vmulq_f32(xv1, av));
        vst1q_f32(x.add(i + 8), vmulq_f32(xv2, av));
        vst1q_f32(x.add(i + 12), vmulq_f32(xv3, av));
        i += 16;
    }
    while i + 4 <= n {
        let xv = vld1q_f32(x.add(i));
        let out = vmulq_f32(xv, av);
        vst1q_f32(x.add(i), out);
        i += 4;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2_inplace(x: *mut f32, alpha: f32, n: usize) {
    let mut i = 0usize;
    let av = _mm256_set1_ps(alpha);
    while i + 16 <= n {
        let xv0 = _mm256_loadu_ps(x.add(i));
        let xv1 = _mm256_loadu_ps(x.add(i + 8));
        _mm256_storeu_ps(x.add(i), _mm256_mul_ps(xv0, av));
        _mm256_storeu_ps(x.add(i + 8), _mm256_mul_ps(xv1, av));
        i += 16;
    }
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(x.add(i), _mm256_mul_ps(xv, av));
        i += 8;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scale_simd_inplace(x: *mut f32, alpha: f32, n: usize) {
    if use_x86_avx2_fma() {
        return scale_avx2_inplace(x, alpha, n);
    }
    let mut i = 0usize;
    let av = _mm_set1_ps(alpha);
    while i + 4 <= n {
        let xv = _mm_loadu_ps(x.add(i));
        let out = _mm_mul_ps(xv, av);
        _mm_storeu_ps(x.add(i), out);
        i += 4;
    }
    while i < n {
        *x.add(i) *= alpha;
        i += 1;
    }
}

#[inline(always)]
fn scale_slice_inplace(x: &mut [f32], alpha: f32) {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        scale_simd_inplace(x.as_mut_ptr(), alpha, x.len());
        return;
    }
    #[allow(unreachable_code)]
    for v in x {
        *v *= alpha;
    }
}

#[inline(always)]
fn vec_dot_f32(x: &[f32], w: &[u8], n: usize) -> f32 {
    let w_ptr = w.as_ptr() as *const f32;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    unsafe {
        return dot_f32_simd_ptr(x.as_ptr(), w_ptr, n);
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i] * read_f32_le(w, i * 4);
        }
        sum
    }
}

#[inline(always)]
fn vec_dot_f16(x: &[f32], w: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = [
                fp16_to_fp32(read_u16_le(w, i * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = vld1q_f32(wv.as_ptr());
            acc = vfmaq_f32(acc, xv, wq);
            i += 4;
        }
        sum += vaddvq_f32(acc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if use_x86_f16c() {
            let n8 = n & !7;
            if n8 > 0 {
                sum += vec_dot_f16_f16c_prefix(x.as_ptr(), w.as_ptr(), n8);
                i = n8;
            }
        }
        let mut acc = _mm_setzero_ps();
        while i + 4 <= n {
            let xv = _mm_loadu_ps(x.as_ptr().add(i));
            let wv = [
                fp16_to_fp32(read_u16_le(w, i * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                fp16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = _mm_loadu_ps(wv.as_ptr());
            acc = _mm_add_ps(acc, _mm_mul_ps(xv, wq));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    while i < n {
        sum += x[i] * fp16_to_fp32(read_u16_le(w, i * 2));
        i += 1;
    }
    sum
}

#[inline(always)]
fn vec_dot_bf16(x: &[f32], w: &[u8], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = [
                bf16_to_fp32(read_u16_le(w, i * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = vld1q_f32(wv.as_ptr());
            acc = vfmaq_f32(acc, xv, wq);
            i += 4;
        }
        sum += vaddvq_f32(acc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if use_x86_avx2_fma() {
            let n8 = n & !7;
            if n8 > 0 {
                sum += vec_dot_bf16_avx2_prefix(x.as_ptr(), w.as_ptr(), n8);
                i = n8;
            }
        }
        let mut acc = _mm_setzero_ps();
        while i + 4 <= n {
            let xv = _mm_loadu_ps(x.as_ptr().add(i));
            let wv = [
                bf16_to_fp32(read_u16_le(w, i * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 1) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 2) * 2)),
                bf16_to_fp32(read_u16_le(w, (i + 3) * 2)),
            ];
            let wq = _mm_loadu_ps(wv.as_ptr());
            acc = _mm_add_ps(acc, _mm_mul_ps(xv, wq));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    while i < n {
        sum += x[i] * bf16_to_fp32(read_u16_le(w, i * 2));
        i += 1;
    }
    sum
}

fn vec_dot_q4_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qs = &w[off + 2..off + 2 + QK4_0 / 2];
        let xb = &x[i * QK4_0..(i + 1) * QK4_0];
        let mut block_sum = 0.0;
        for j in 0..QK4_0 / 2 {
            let x0 = (qs[j] & 0x0f) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            block_sum += xb[j] * x0 as f32 + xb[j + QK4_0 / 2] * x1 as f32;
        }
        sum += block_sum * d;
    }
    sum
}

fn vec_dot_q4_1(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_1));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let m = fp16_to_fp32(read_u16_le(w, off + 2));
        let qs = &w[off + 4..off + 4 + QK4_1 / 2];
        let xb = &x[i * QK4_1..(i + 1) * QK4_1];
        let mut block_sum = 0.0;
        let mut x_sum = 0.0;
        for j in 0..QK4_1 / 2 {
            let x0 = (qs[j] & 0x0f) as f32;
            let x1 = (qs[j] >> 4) as f32;
            block_sum += xb[j] * x0 + xb[j + QK4_1 / 2] * x1;
            x_sum += xb[j] + xb[j + QK4_1 / 2];
        }
        sum += block_sum * d + x_sum * m;
    }
    sum
}

fn vec_dot_q5_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK5_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qh = read_u32_le(w, off + 2);
        let qs = &w[off + 6..off + 6 + QK5_0 / 2];
        let xb = &x[i * QK5_0..(i + 1) * QK5_0];
        let mut block_sum = 0.0;
        for j in 0..QK5_0 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
            block_sum += xb[j] * x0 as f32 + xb[j + QK5_0 / 2] * x1 as f32;
        }
        sum += block_sum * d;
    }
    sum
}

fn vec_dot_q5_1(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK5_1;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_1));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let m = fp16_to_fp32(read_u16_le(w, off + 2));
        let qh = read_u32_le(w, off + 4);
        let qs = &w[off + 8..off + 8 + QK5_1 / 2];
        let xb = &x[i * QK5_1..(i + 1) * QK5_1];
        let mut block_sum = 0.0;
        let mut x_sum = 0.0;
        for j in 0..QK5_1 / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0f) as u32 | xh0) as f32;
            let x1 = ((qs[j] >> 4) as u32 | xh1) as f32;
            block_sum += xb[j] * x0 + xb[j + QK5_1 / 2] * x1;
            x_sum += xb[j] + xb[j + QK5_1 / 2];
        }
        sum += block_sum * d + x_sum * m;
    }
    sum
}

fn vec_dot_q8_0(x: &[f32], w: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if use_aarch64_dotprod_q8() {
        unsafe {
            return vec_dot_q8_0_dotprod(x, w, n);
        }
    }

    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut qf = [0.0f32; QK8_0];
        for j in 0..QK8_0 {
            qf[j] = w[off + 2 + j] as i8 as f32;
        }
        let block_sum = dot_f32_simd(xb, &qf);
        sum += block_sum * d;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_16_neon(a: *const i8, b: *const i8) -> i32 {
    let av = vld1q_s8(a);
    let bv = vld1q_s8(b);
    let prod0 = vmull_s8(vget_low_s8(av), vget_low_s8(bv));
    let prod1 = vmull_s8(vget_high_s8(av), vget_high_s8(bv));
    let sum0 = vpaddlq_s16(prod0);
    let sum1 = vpaddlq_s16(prod1);
    vaddvq_s32(vaddq_s32(sum0, sum1))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_32_dotprod(a: *const i8, b: *const i8) -> i32 {
    dot_i8_16_neon(a, b) + dot_i8_16_neon(a.add(16), b.add(16))
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_dot_q8_0_dotprod(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK8_0;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q8_0));
    let mut sum = 0.0f32;
    let mut xq = [0i8; QK8_0];

    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let xb = &x[i * QK8_0..(i + 1) * QK8_0];
        let mut abs_max = 0.0f32;
        for &v in xb {
            let a = v.abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0.0 {
            continue;
        }
        let x_scale = abs_max / 127.0;
        let inv_x_scale = 1.0 / x_scale;
        for j in 0..QK8_0 {
            let q = (xb[j] * inv_x_scale).round().clamp(-127.0, 127.0);
            xq[j] = q as i8;
        }
        let q_ptr = w[off + 2..off + 2 + QK8_0].as_ptr() as *const i8;
        let dot_i32 = dot_i8_32_dotprod(xq.as_ptr(), q_ptr);
        sum += dot_i32 as f32 * x_scale * d;
    }
    sum
}

fn vec_dot_q2_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q2_K));
    let mut sum = 0.0;

    for i in 0..nb {
        let off = i * block_sz;
        let scales = &w[off..off + QK_K / 16];
        let mut q_off = off + QK_K / 16;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 16 + QK_K / 4));
        let dmin = fp16_to_fp32(read_u16_le(w, off + QK_K / 16 + QK_K / 4 + 2));
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0;

        for n_outer in (0..QK_K).step_by(128) {
            let q = &w[q_off..q_off + 32];
            let mut shift = 0;
            for j in 0..4 {
                let sc = scales[is];
                is += 1;
                let mut dl = d * (sc & 0x0f) as f32;
                let mut ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + l;
                    let wv = dl * ((q[l] >> shift) & 0x03) as f32 - ml;
                    block_sum += xb[idx] * wv;
                }
                let sc2 = scales[is];
                is += 1;
                dl = d * (sc2 & 0x0f) as f32;
                ml = dmin * (sc2 >> 4) as f32;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + 16 + l;
                    let wv = dl * ((q[l + 16] >> shift) & 0x03) as f32 - ml;
                    block_sum += xb[idx] * wv;
                }
                shift += 2;
            }
            q_off += 32;
        }
        sum += block_sum;
    }
    sum
}

fn vec_dot_q3_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q3_K));
    let mut sum = 0.0;

    for i in 0..nb {
        let off = i * block_sz;
        let hmask = &w[off..off + QK_K / 8];
        let mut q_off = off + QK_K / 8;
        let scales = q3_scales(&w[off + QK_K / 8 + QK_K / 4..off + QK_K / 8 + QK_K / 4 + 12]);
        let d_all = fp16_to_fp32(read_u16_le(w, off + QK_K / 8 + QK_K / 4 + 12));
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut block_sum = 0.0;

        for n_outer in (0..QK_K).step_by(128) {
            let q = &w[q_off..q_off + 32];
            let mut shift = 0usize;
            for j in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + l;
                    let wv = ((q[l] >> shift) & 3) as i8 - if (hmask[l] & m) != 0 { 0 } else { 4 };
                    block_sum += xb[idx] * dl * wv as f32;
                }
                let dl2 = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let idx = n_outer + j * 32 + 16 + l;
                    let wv = ((q[l + 16] >> shift) & 3) as i8
                        - if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    block_sum += xb[idx] * dl2 * wv as f32;
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
        sum += block_sum;
    }
    sum
}

fn vec_dot_q4_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let mut q_off = off + 16;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0f32;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &w[q_off..q_off + 32];
            for l in 0..32 {
                let qv = q[l];
                let w0 = d1 * (qv & 0x0f) as f32 - m1f;
                let w1 = d2 * (qv >> 4) as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            q_off += 32;
            is += 2;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let mut q_off = off + 16;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut block_sum = 0.0;

        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let q = &w[q_off..q_off + 32];
            for l in 0..32 {
                let qv = q[l];
                let w0 = d1 * (qv & 0x0f) as f32 - m1f;
                let w1 = d2 * (qv >> 4) as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            q_off += 32;
            is += 2;
        }
        sum += block_sum;
    }
    sum
}

fn vec_dot_q5_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let qh = &w[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut block_sum = 0.0f32;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;
            let ql = &w[ql_off..ql_off + 32];
            for l in 0..32 {
                let qv = ql[l];
                let lo = (qv & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                let w0 = d1 * lo as f32 - m1f;
                let w1 = d2 * hi as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let dmin = fp16_to_fp32(read_u16_le(w, off + 2));
        let scales = &w[off + 4..off + 16];
        let qh = &w[off + 16..off + 16 + QK_K / 8];
        let mut ql_off = off + 16 + QK_K / 8;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut block_sum = 0.0;

        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2f = dmin * m2 as f32;

            let ql = &w[ql_off..ql_off + 32];

            for l in 0..32 {
                let qv = ql[l];
                let lo = (qv & 0x0f) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi = (qv >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                let w0 = d1 * lo as f32 - m1f;
                let w1 = d2 * hi as f32 - m2f;
                block_sum += xb[j + l] * w0 + xb[j + 32 + l] * w1;
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += block_sum;
    }
    sum
}

fn vec_dot_q6_k(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sum = 0.0;

    #[cfg(target_arch = "aarch64")]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut block_sum = 0.0f32;
        for n_outer in (0..QK_K).step_by(128) {
            let ql = &w[ql_off..ql_off + 64];
            let qh = &w[qh_off..qh_off + 32];
            let sc = &w[sc_off..sc_off + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                let s0 = d * sc[is] as i8 as f32;
                let s1 = d * sc[is + 2] as i8 as f32;
                let s2 = d * sc[is + 4] as i8 as f32;
                let s3 = d * sc[is + 6] as i8 as f32;
                block_sum += xb[n_outer + l] * (s0 * q1 as f32);
                block_sum += xb[n_outer + 32 + l] * (s1 * q2 as f32);
                block_sum += xb[n_outer + 64 + l] * (s2 * q3 as f32);
                block_sum += xb[n_outer + 96 + l] * (s3 * q4 as f32);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        sum += block_sum;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off + QK_K / 2 + QK_K / 4 + QK_K / 16));
        let mut ql_off = off;
        let mut qh_off = off + QK_K / 2;
        let mut sc_off = off + QK_K / 2 + QK_K / 4;
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut block_sum = 0.0;
        for n_outer in (0..QK_K).step_by(128) {
            let ql = &w[ql_off..ql_off + 64];
            let qh = &w[qh_off..qh_off + 32];
            let sc = &w[sc_off..sc_off + 8];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;
                let s0 = d * sc[is] as i8 as f32;
                let s1 = d * sc[is + 2] as i8 as f32;
                let s2 = d * sc[is + 4] as i8 as f32;
                let s3 = d * sc[is + 6] as i8 as f32;
                block_sum += xb[n_outer + l] * (s0 * q1 as f32);
                block_sum += xb[n_outer + 32 + l] * (s1 * q2 as f32);
                block_sum += xb[n_outer + 64 + l] * (s2 * q3 as f32);
                block_sum += xb[n_outer + 96 + l] * (s3 * q4 as f32);
            }

            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        sum += block_sum;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vec_dot_q4_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            q_off[r] = off + 16;
        }

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][q_off[r] + l];
                    sums[r] += x0 * (a_lo[r] * (qv & 0x0f) as f32 - b_lo[r])
                        + x1 * (a_hi[r] * (qv >> 4) as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vec_dot_q5_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut qh = [&[][..]; 4];
        let mut ql_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            qh[r] = &rows[r][off + 16..off + 16 + QK_K / 8];
            ql_off[r] = off + 16 + QK_K / 8;
        }

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][ql_off[r] + l];
                    let lo = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    let hi = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                    sums[r] += x0 * (a_lo[r] * lo as f32 - b_lo[r])
                        + x1 * (a_hi[r] * hi as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vec_dot_q6_k_4rows(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut ql_off = [0usize; 4];
        let mut qh_off = [0usize; 4];
        let mut sc_off = [0usize; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(
                rows[r],
                off + QK_K / 2 + QK_K / 4 + QK_K / 16,
            ));
            ql_off[r] = off;
            qh_off[r] = off + QK_K / 2;
            sc_off[r] = off + QK_K / 2 + QK_K / 4;
        }

        for n_outer in (0..QK_K).step_by(128) {
            let mut ql = [&[][..]; 4];
            let mut qh = [&[][..]; 4];
            let mut sc = [&[][..]; 4];
            for r in 0..4 {
                ql[r] = &rows[r][ql_off[r]..ql_off[r] + 64];
                qh[r] = &rows[r][qh_off[r]..qh_off[r] + 32];
                sc[r] = &rows[r][sc_off[r]..sc_off[r] + 8];
            }

            for l in 0..32 {
                let is = l / 16;
                let x0 = xb[n_outer + l];
                let x1 = xb[n_outer + 32 + l];
                let x2 = xb[n_outer + 64 + l];
                let x3 = xb[n_outer + 96 + l];
                for r in 0..4 {
                    let ql0 = ql[r][l];
                    let ql1 = ql[r][l + 32];
                    let qh0 = qh[r][l];
                    let q1 = (((ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4)) as i8) - 32;
                    let q2 = (((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) as i8) - 32;
                    let q3 = (((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) as i8) - 32;
                    let q4 = (((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) as i8) - 32;
                    let s0 = d[r] * sc[r][is] as i8 as f32;
                    let s1 = d[r] * sc[r][is + 2] as i8 as f32;
                    let s2 = d[r] * sc[r][is + 4] as i8 as f32;
                    let s3 = d[r] * sc[r][is + 6] as i8 as f32;
                    sums[r] += x0 * (s0 * q1 as f32)
                        + x1 * (s1 * q2 as f32)
                        + x2 * (s2 * q3 as f32)
                        + x3 * (s3 * q4 as f32);
                }
            }
            for r in 0..4 {
                ql_off[r] += 64;
                qh_off[r] += 32;
                sc_off[r] += 8;
            }
        }
    }
    sums
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn matmul_qk_mr4_chunk(
    out: &mut [f32],
    base_row: usize,
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) {
    let mut i = 0usize;
    while i + 4 <= out.len() {
        let row0_off = data_offset + (base_row + i) * row_size;
        let row1_off = row0_off + row_size;
        let row2_off = row1_off + row_size;
        let row3_off = row2_off + row_size;
        let r0 = &mapped[row0_off..row0_off + row_size];
        let r1 = &mapped[row1_off..row1_off + row_size];
        let r2 = &mapped[row2_off..row2_off + row_size];
        let r3 = &mapped[row3_off..row3_off + row_size];
        let sums = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        out[i] = sums[0];
        out[i + 1] = sums[1];
        out[i + 2] = sums[2];
        out[i + 3] = sums[3];
        i += 4;
    }
    while i < out.len() {
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn mr4_status(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_K => &AARCH64_Q4K_MR4_STATUS,
        GGML_TYPE_Q5_K => &AARCH64_Q5K_MR4_STATUS,
        GGML_TYPE_Q6_K => &AARCH64_Q6K_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn validate_qk_mr4_once(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = mr4_status(ttype);
    match status.load(AtomicOrdering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }

    let r0 = &mapped[data_offset..data_offset + row_size];
    let r1 = &mapped[data_offset + row_size..data_offset + 2 * row_size];
    let r2 = &mapped[data_offset + 2 * row_size..data_offset + 3 * row_size];
    let r3 = &mapped[data_offset + 3 * row_size..data_offset + 4 * row_size];

    let mr4 = match ttype {
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_K => [
            vec_dot_q4_k(x, r0, n),
            vec_dot_q4_k(x, r1, n),
            vec_dot_q4_k(x, r2, n),
            vec_dot_q4_k(x, r3, n),
        ],
        GGML_TYPE_Q5_K => [
            vec_dot_q5_k(x, r0, n),
            vec_dot_q5_k(x, r1, n),
            vec_dot_q5_k(x, r2, n),
            vec_dot_q5_k(x, r3, n),
        ],
        GGML_TYPE_Q6_K => [
            vec_dot_q6_k(x, r0, n),
            vec_dot_q6_k(x, r1, n),
            vec_dot_q6_k(x, r2, n),
            vec_dot_q6_k(x, r3, n),
        ],
        _ => unreachable!(),
    };

    let mut ok = true;
    for i in 0..4 {
        let a = mr4[i];
        let b = scalar[i];
        let tol = 1e-4f32 * b.abs().max(1.0);
        if (a - b).abs() > tol {
            ok = false;
            break;
        }
    }

    status.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
    if !ok {
        eprintln!(
            "Warning: disabling aarch64 MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn try_matmul_qk_mr4(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    if !use_aarch64_qk_mr4() {
        return false;
    }
    if !matches!(ttype, GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K) {
        return false;
    }
    if n < QK_K || n % QK_K != 0 {
        return false;
    }

    let d = xout.len();
    if d < 4 {
        return false;
    }
    if !validate_qk_mr4_once(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }
    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_qk_mr4_chunk(chunk, base_row, x, mapped, data_offset, row_size, n, ttype);
            });
    } else {
        matmul_qk_mr4_chunk(xout, 0, x, mapped, data_offset, row_size, n, ttype);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn vec_dot_q4_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q4_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut q_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            q_off[r] = off + 16;
        }

        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][q_off[r] + l];
                    sums[r] += x0 * (a_lo[r] * (qv & 0x0f) as f32 - b_lo[r])
                        + x1 * (a_hi[r] * (qv >> 4) as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                q_off[r] += 32;
            }
            is += 2;
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn vec_dot_q5_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q5_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut dmin = [0.0f32; 4];
        let mut scales = [&[][..]; 4];
        let mut qh = [&[][..]; 4];
        let mut ql_off = [0usize; 4];

        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(rows[r], off));
            dmin[r] = fp16_to_fp32(read_u16_le(rows[r], off + 2));
            scales[r] = &rows[r][off + 4..off + 16];
            qh[r] = &rows[r][off + 16..off + 16 + QK_K / 8];
            ql_off[r] = off + 16 + QK_K / 8;
        }

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let mut a_lo = [0.0f32; 4];
            let mut b_lo = [0.0f32; 4];
            let mut a_hi = [0.0f32; 4];
            let mut b_hi = [0.0f32; 4];
            for r in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, scales[r]);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales[r]);
                a_lo[r] = d[r] * sc1 as f32;
                b_lo[r] = dmin[r] * m1 as f32;
                a_hi[r] = d[r] * sc2 as f32;
                b_hi[r] = dmin[r] * m2 as f32;
            }
            for l in 0..32 {
                let x0 = xb[j + l];
                let x1 = xb[j + 32 + l];
                for r in 0..4 {
                    let qv = rows[r][ql_off[r] + l];
                    let lo = (qv & 0x0f) + if (qh[r][l] & u1) != 0 { 16 } else { 0 };
                    let hi = (qv >> 4) + if (qh[r][l] & u2) != 0 { 16 } else { 0 };
                    sums[r] += x0 * (a_lo[r] * lo as f32 - b_lo[r])
                        + x1 * (a_hi[r] * hi as f32 - b_hi[r]);
                }
            }
            for r in 0..4 {
                ql_off[r] += 32;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn vec_dot_q6_k_4rows_x86(
    x: &[f32],
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    n: usize,
) -> [f32; 4] {
    let rows = [r0, r1, r2, r3];
    let nb = n / QK_K;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_Q6_K));
    let mut sums = [0.0f32; 4];

    for i in 0..nb {
        let off = i * block_sz;
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut d = [0.0f32; 4];
        let mut ql_off = [0usize; 4];
        let mut qh_off = [0usize; 4];
        let mut sc_off = [0usize; 4];
        for r in 0..4 {
            d[r] = fp16_to_fp32(read_u16_le(
                rows[r],
                off + QK_K / 2 + QK_K / 4 + QK_K / 16,
            ));
            ql_off[r] = off;
            qh_off[r] = off + QK_K / 2;
            sc_off[r] = off + QK_K / 2 + QK_K / 4;
        }

        for n_outer in (0..QK_K).step_by(128) {
            let mut ql = [&[][..]; 4];
            let mut qh = [&[][..]; 4];
            let mut sc = [&[][..]; 4];
            for r in 0..4 {
                ql[r] = &rows[r][ql_off[r]..ql_off[r] + 64];
                qh[r] = &rows[r][qh_off[r]..qh_off[r] + 32];
                sc[r] = &rows[r][sc_off[r]..sc_off[r] + 8];
            }

            for l in 0..32 {
                let is = l / 16;
                let x0 = xb[n_outer + l];
                let x1 = xb[n_outer + 32 + l];
                let x2 = xb[n_outer + 64 + l];
                let x3 = xb[n_outer + 96 + l];
                for r in 0..4 {
                    let ql0 = ql[r][l];
                    let ql1 = ql[r][l + 32];
                    let qh0 = qh[r][l];
                    let q1 = (((ql0 & 0x0f) | (((qh0 >> 0) & 0x03) << 4)) as i8) - 32;
                    let q2 = (((ql1 & 0x0f) | (((qh0 >> 2) & 0x03) << 4)) as i8) - 32;
                    let q3 = (((ql0 >> 4) | (((qh0 >> 4) & 0x03) << 4)) as i8) - 32;
                    let q4 = (((ql1 >> 4) | (((qh0 >> 6) & 0x03) << 4)) as i8) - 32;
                    let s0 = d[r] * sc[r][is] as i8 as f32;
                    let s1 = d[r] * sc[r][is + 2] as i8 as f32;
                    let s2 = d[r] * sc[r][is + 4] as i8 as f32;
                    let s3 = d[r] * sc[r][is + 6] as i8 as f32;
                    sums[r] += x0 * (s0 * q1 as f32)
                        + x1 * (s1 * q2 as f32)
                        + x2 * (s2 * q3 as f32)
                        + x3 * (s3 * q4 as f32);
                }
            }
            for r in 0..4 {
                ql_off[r] += 64;
                qh_off[r] += 32;
                sc_off[r] += 8;
            }
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn matmul_qk_mr4_chunk_x86(
    out: &mut [f32],
    base_row: usize,
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) {
    let mut i = 0usize;
    while i + 4 <= out.len() {
        let row0_off = data_offset + (base_row + i) * row_size;
        let row1_off = row0_off + row_size;
        let row2_off = row1_off + row_size;
        let row3_off = row2_off + row_size;
        let r0 = &mapped[row0_off..row0_off + row_size];
        let r1 = &mapped[row1_off..row1_off + row_size];
        let r2 = &mapped[row2_off..row2_off + row_size];
        let r3 = &mapped[row3_off..row3_off + row_size];
        let sums = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
            _ => unreachable!(),
        };
        out[i] = sums[0];
        out[i + 1] = sums[1];
        out[i + 2] = sums[2];
        out[i + 3] = sums[3];
        i += 4;
    }
    while i < out.len() {
        let row_off = data_offset + (base_row + i) * row_size;
        let row = &mapped[row_off..row_off + row_size];
        out[i] = match ttype {
            GGML_TYPE_Q4_K => vec_dot_q4_k(x, row, n),
            GGML_TYPE_Q5_K => vec_dot_q5_k(x, row, n),
            GGML_TYPE_Q6_K => vec_dot_q6_k(x, row, n),
            _ => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn mr4_status_x86(ttype: i32) -> &'static AtomicU8 {
    match ttype {
        GGML_TYPE_Q4_K => &X86_Q4K_MR4_STATUS,
        GGML_TYPE_Q5_K => &X86_Q5K_MR4_STATUS,
        GGML_TYPE_Q6_K => &X86_Q6K_MR4_STATUS,
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn validate_qk_mr4_once_x86(
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    let status = mr4_status_x86(ttype);
    match status.load(AtomicOrdering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }

    let r0 = &mapped[data_offset..data_offset + row_size];
    let r1 = &mapped[data_offset + row_size..data_offset + 2 * row_size];
    let r2 = &mapped[data_offset + 2 * row_size..data_offset + 3 * row_size];
    let r3 = &mapped[data_offset + 3 * row_size..data_offset + 4 * row_size];

    let mr4 = match ttype {
        GGML_TYPE_Q4_K => vec_dot_q4_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q5_K => vec_dot_q5_k_4rows_x86(x, r0, r1, r2, r3, n),
        GGML_TYPE_Q6_K => vec_dot_q6_k_4rows_x86(x, r0, r1, r2, r3, n),
        _ => unreachable!(),
    };
    let scalar = match ttype {
        GGML_TYPE_Q4_K => [
            vec_dot_q4_k(x, r0, n),
            vec_dot_q4_k(x, r1, n),
            vec_dot_q4_k(x, r2, n),
            vec_dot_q4_k(x, r3, n),
        ],
        GGML_TYPE_Q5_K => [
            vec_dot_q5_k(x, r0, n),
            vec_dot_q5_k(x, r1, n),
            vec_dot_q5_k(x, r2, n),
            vec_dot_q5_k(x, r3, n),
        ],
        GGML_TYPE_Q6_K => [
            vec_dot_q6_k(x, r0, n),
            vec_dot_q6_k(x, r1, n),
            vec_dot_q6_k(x, r2, n),
            vec_dot_q6_k(x, r3, n),
        ],
        _ => unreachable!(),
    };

    let mut ok = true;
    for i in 0..4 {
        let a = mr4[i];
        let b = scalar[i];
        let tol = 1e-4f32 * b.abs().max(1.0);
        if (a - b).abs() > tol {
            ok = false;
            break;
        }
    }

    status.store(if ok { 1 } else { 2 }, AtomicOrdering::Relaxed);
    if !ok {
        eprintln!(
            "Warning: disabling x86_64 MR4 kernel for type {} due to validation mismatch",
            ttype
        );
    }
    ok
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn try_matmul_qk_mr4_x86(
    xout: &mut [f32],
    x: &[f32],
    mapped: &[u8],
    data_offset: usize,
    row_size: usize,
    n: usize,
    ttype: i32,
) -> bool {
    if !use_x86_qk_mr4() {
        return false;
    }
    if !matches!(ttype, GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K) {
        return false;
    }
    if n < QK_K || n % QK_K != 0 {
        return false;
    }

    let d = xout.len();
    if d < 4 {
        return false;
    }
    if !validate_qk_mr4_once_x86(x, mapped, data_offset, row_size, n, ttype) {
        return false;
    }
    let chunk_rows = par_matmul_chunk_rows();
    if d >= par_matmul_min_rows() {
        xout.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_row = chunk_idx * chunk_rows;
                matmul_qk_mr4_chunk_x86(chunk, base_row, x, mapped, data_offset, row_size, n, ttype);
            });
    } else {
        matmul_qk_mr4_chunk_x86(xout, 0, x, mapped, data_offset, row_size, n, ttype);
    }
    true
}

fn vec_dot_iq4_nl(x: &[f32], w: &[u8], n: usize) -> f32 {
    let nb = n / QK4_NL;
    let block_sz = get_type_size(GgmlType(GGML_TYPE_IQ4_NL));
    let mut sum = 0.0;
    for i in 0..nb {
        let off = i * block_sz;
        let d = fp16_to_fp32(read_u16_le(w, off));
        let qs = &w[off + 2..off + 2 + QK4_NL / 2];
        let xb = &x[i * QK4_NL..(i + 1) * QK4_NL];
        let mut block_sum = 0.0;
        for j in 0..QK4_NL / 2 {
            block_sum += xb[j] * KVALUES_IQ4NL[(qs[j] & 0x0f) as usize] as f32;
            block_sum += xb[j + QK4_NL / 2] * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
        sum += block_sum * d;
    }
    sum
}

fn get_row_size(n_cols: usize, ttype: GgmlType) -> usize {
    let block_size = get_block_size(ttype);
    let type_size = get_type_size(ttype);
    (n_cols / block_size) * type_size
}

fn matmul_quantized(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    mapped: &[u8],
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d = qw.rows;
    let n = qw.cols;
    let row_size = get_row_size(n, qw.ttype);
    if xout.len() < d || x.len() < n {
        return Err("matmul shape mismatch".to_string());
    }
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = qw
        .data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    let data_offset = qw.data_offset;
    ensure_model_range(data_offset, data_size)?;
    macro_rules! run_rows {
        ($dot:path) => {{
            if d >= par_matmul_min_rows() {
                let chunk_rows = par_matmul_chunk_rows();
                xout[..d]
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_row = chunk_idx * chunk_rows;
                        for (j, out) in out_chunk.iter_mut().enumerate() {
                            let row_off = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_off..row_off + row_size];
                            *out = $dot(x, row, n);
                        }
                    });
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    let row_off = data_offset + i * row_size;
                    let row = &mapped[row_off..row_off + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => run_rows!(vec_dot_q4_0),
        GGML_TYPE_Q4_1 => run_rows!(vec_dot_q4_1),
        GGML_TYPE_Q5_0 => run_rows!(vec_dot_q5_0),
        GGML_TYPE_Q5_1 => run_rows!(vec_dot_q5_1),
        GGML_TYPE_Q8_0 => run_rows!(vec_dot_q8_0),
        GGML_TYPE_Q2_K => run_rows!(vec_dot_q2_k),
        GGML_TYPE_Q3_K => run_rows!(vec_dot_q3_k),
        GGML_TYPE_Q4_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_k);
            }
        }
        GGML_TYPE_Q5_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_k);
            }
        }
        GGML_TYPE_Q6_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => run_rows!(vec_dot_iq4_nl),
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 | 30 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ))
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

fn matmul_quantized_rows(
    xout: &mut [f32],
    x: &[f32],
    qw: &QuantizedTensor,
    row_start: usize,
    n_rows: usize,
    mapped: &[u8],
) -> Result<(), String> {
    let prof_t0 = prof_start();
    let d = n_rows;
    let n = qw.cols;
    let row_size = get_row_size(n, qw.ttype);
    if row_start + n_rows > qw.rows {
        return Err("matmul row window exceeds tensor rows".to_string());
    }
    if xout.len() < d || x.len() < n {
        return Err("matmul shape mismatch".to_string());
    }
    let row_off = row_start
        .checked_mul(row_size)
        .ok_or_else(|| "quantized row offset overflow".to_string())?;
    let data_offset = qw
        .data_offset
        .checked_add(row_off)
        .ok_or_else(|| "quantized tensor offset overflow".to_string())?;
    let data_size = d
        .checked_mul(row_size)
        .ok_or_else(|| "quantized tensor row size overflow".to_string())?;
    let data_end = data_offset
        .checked_add(data_size)
        .ok_or_else(|| "quantized tensor end overflow".to_string())?;
    if data_end > mapped.len() {
        return Err("quantized row outside mapped file".to_string());
    }
    ensure_model_range(data_offset, data_size)?;
    macro_rules! run_rows {
        ($dot:path) => {{
            if d >= par_matmul_min_rows() {
                let chunk_rows = par_matmul_chunk_rows();
                xout[..d]
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_row = chunk_idx * chunk_rows;
                        for (j, out) in out_chunk.iter_mut().enumerate() {
                            let row_start = data_offset + (base_row + j) * row_size;
                            let row = &mapped[row_start..row_start + row_size];
                            *out = $dot(x, row, n);
                        }
                    });
            } else {
                for (i, out) in xout[..d].iter_mut().enumerate() {
                    let row_start = data_offset + i * row_size;
                    let row = &mapped[row_start..row_start + row_size];
                    *out = $dot(x, row, n);
                }
            }
        }};
    }

    match qw.ttype.0 {
        GGML_TYPE_Q4_0 => run_rows!(vec_dot_q4_0),
        GGML_TYPE_Q4_1 => run_rows!(vec_dot_q4_1),
        GGML_TYPE_Q5_0 => run_rows!(vec_dot_q5_0),
        GGML_TYPE_Q5_1 => run_rows!(vec_dot_q5_1),
        GGML_TYPE_Q8_0 => run_rows!(vec_dot_q8_0),
        GGML_TYPE_Q2_K => run_rows!(vec_dot_q2_k),
        GGML_TYPE_Q3_K => run_rows!(vec_dot_q3_k),
        GGML_TYPE_Q4_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q4_K,
                ) {
                    run_rows!(vec_dot_q4_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q4_k);
            }
        }
        GGML_TYPE_Q5_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q5_K,
                ) {
                    run_rows!(vec_dot_q5_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q5_k);
            }
        }
        GGML_TYPE_Q6_K => {
            #[cfg(target_arch = "aarch64")]
            {
                if !try_matmul_qk_mr4(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if !try_matmul_qk_mr4_x86(
                    &mut xout[..d],
                    x,
                    mapped,
                    data_offset,
                    row_size,
                    n,
                    GGML_TYPE_Q6_K,
                ) {
                    run_rows!(vec_dot_q6_k);
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                run_rows!(vec_dot_q6_k);
            }
        }
        GGML_TYPE_IQ4_NL => run_rows!(vec_dot_iq4_nl),
        GGML_TYPE_F16 => run_rows!(vec_dot_f16),
        GGML_TYPE_BF16 | 30 => run_rows!(vec_dot_bf16),
        GGML_TYPE_F32 => run_rows!(vec_dot_f32),
        _ => {
            return Err(format!(
                "unsupported quantization type in matmul: {}",
                qw.ttype.0
            ))
        }
    }

    prof_end(&PROF_MATMUL_NS, prof_t0);
    Ok(())
}

fn select_topk_softmax(
    logits: &[f32],
    k: usize,
    n_group: usize,
    topk_group: usize,
    normalize_topk: bool,
    scale: f32,
    scores_scratch: &mut Vec<f32>,
    selected_group_scratch: &mut Vec<bool>,
    group_scores_scratch: &mut Vec<f32>,
    rank_scratch: &mut Vec<usize>,
    out_indices: &mut [usize],
    out_weights: &mut [f32],
) -> usize {
    let top_k = k.max(1).min(logits.len());
    if scores_scratch.len() < logits.len() {
        scores_scratch.resize(logits.len(), 0.0);
    }
    let scores = &mut scores_scratch[..logits.len()];
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max_logit).exp();
        scores[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum.max(f32::MIN_POSITIVE);
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }

    let use_grouped = n_group > 1 && topk_group < n_group && logits.len() % n_group == 0;
    let group_size = if use_grouped {
        logits.len() / n_group
    } else {
        logits.len()
    };

    let selected_group_len = n_group.max(1);
    if selected_group_scratch.len() < selected_group_len {
        selected_group_scratch.resize(selected_group_len, true);
    }
    let selected_group = &mut selected_group_scratch[..selected_group_len];
    selected_group.fill(true);

    if use_grouped {
        if group_scores_scratch.len() < n_group {
            group_scores_scratch.resize(n_group, 0.0);
        }
        let group_scores = &mut group_scores_scratch[..n_group];
        for g in 0..n_group {
            let start = g * group_size;
            let end = start + group_size;
            let mut best1 = f32::NEG_INFINITY;
            let mut best2 = f32::NEG_INFINITY;
            for &s in &scores[start..end] {
                if s > best1 {
                    best2 = best1;
                    best1 = s;
                } else if s > best2 {
                    best2 = s;
                }
            }
            group_scores[g] = best1 + if best2.is_finite() { best2 } else { 0.0 };
        }

        selected_group.fill(false);
        if rank_scratch.len() < n_group {
            rank_scratch.resize(n_group, 0);
        }
        let rank = &mut rank_scratch[..n_group];
        for (i, r) in rank.iter_mut().enumerate() {
            *r = i;
        }
        rank.sort_by(|&a, &b| {
            group_scores[b]
                .partial_cmp(&group_scores[a])
                .unwrap_or(Ordering::Equal)
        });
        for &g in rank.iter().take(topk_group.max(1).min(n_group)) {
            selected_group[g] = true;
        }
    }

    for i in 0..top_k {
        out_weights[i] = f32::NEG_INFINITY;
        out_indices[i] = 0;
    }
    let mut count = 0usize;

    for (idx, &v) in scores.iter().enumerate() {
        if use_grouped {
            let g = idx / group_size;
            if !selected_group[g] {
                continue;
            }
        }
        if count < top_k {
            let mut ins = count;
            while ins > 0 && v > out_weights[ins - 1] {
                out_weights[ins] = out_weights[ins - 1];
                out_indices[ins] = out_indices[ins - 1];
                ins -= 1;
            }
            out_weights[ins] = v;
            out_indices[ins] = idx;
            count += 1;
            continue;
        }

        if v <= out_weights[top_k - 1] {
            continue;
        }

        out_weights[top_k - 1] = v;
        out_indices[top_k - 1] = idx;
        let mut pos = top_k - 1;
        while pos > 0 && out_weights[pos] > out_weights[pos - 1] {
            out_weights.swap(pos, pos - 1);
            out_indices.swap(pos, pos - 1);
            pos -= 1;
        }
    }

    if count == 0 {
        return 0;
    }

    if top_k > 1 && normalize_topk {
        let mut sum_selected = 0.0f32;
        for i in 0..count {
            sum_selected += out_weights[i];
        }
        let inv = 1.0 / sum_selected.max(f32::MIN_POSITIVE);
        for i in 0..count {
            out_weights[i] *= inv;
        }
    }

    for i in 0..count {
        out_weights[i] *= scale;
    }

    count
}

fn tensor_n_elements(tensor: &Gguftensor) -> usize {
    let mut n_elements = 1usize;
    for i in 0..tensor.n_dims as usize {
        n_elements = n_elements.saturating_mul(tensor.ne[i] as usize);
    }
    n_elements
}

fn load_tensor_float(
    gguf: &GGUFFile,
    name: &str,
    expected_elements: Option<usize>,
) -> Result<Vec<f32>, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);

    if let Some(expected) = expected_elements {
        if expected != n_elements {
            eprintln!(
                "Warning: tensor {} has {} elements, expected {}",
                name, n_elements, expected
            );
        }
    }

    let block_size = get_block_size(tensor.ttype);
    let type_size = get_type_size(tensor.ttype);
    if type_size == 0 {
        return Err(format!(
            "unsupported tensor type {} for {name}",
            tensor.ttype.0
        ));
    }

    if n_elements % block_size != 0 {
        return Err(format!(
            "tensor {name} element count {n_elements} not divisible by block size {block_size}"
        ));
    }

    let src_size = (n_elements / block_size) * type_size;
    let mapped = gguf.mapped.as_slice();
    if tensor.data_offset + src_size > mapped.len() {
        return Err(format!("tensor {name} exceeds mapped file bounds"));
    }
    gguf.ensure_range(tensor.data_offset, src_size)?;
    let src = &mapped[tensor.data_offset..tensor.data_offset + src_size];

    dequantize_tensor(src, n_elements, tensor.ttype)
}

fn load_tensor_quantized(
    gguf: &GGUFFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let tensor = find_gguf_tensor(gguf, name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    if n_elements != rows.saturating_mul(cols) {
        return Err(format!(
            "tensor {name} shape mismatch: got {} elements, expected {} (rows={rows}, cols={cols})",
            n_elements,
            rows.saturating_mul(cols)
        ));
    }

    Ok(QuantizedTensor {
        data_offset: tensor.data_offset,
        ttype: tensor.ttype,
        rows,
        cols,
    })
}

fn load_layer_tensor_float(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, String> {
    let name = format!("blk.{layer}.{suffix}");
    load_tensor_float(gguf, &name, Some(expected_elements))
}

fn load_layer_tensor_quantized(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    rows: usize,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let name = format!("blk.{layer}.{suffix}");
    load_tensor_quantized(gguf, &name, rows, cols)
}

fn load_layer_tensor_quantized_auto_rows(
    gguf: &GGUFFile,
    layer: usize,
    suffix: &str,
    cols: usize,
) -> Result<QuantizedTensor, String> {
    let name = format!("blk.{layer}.{suffix}");
    let tensor =
        find_gguf_tensor(gguf, &name).ok_or_else(|| format!("tensor not found: {name}"))?;
    let n_elements = tensor_n_elements(tensor);
    if cols == 0 || n_elements % cols != 0 {
        return Err(format!(
            "tensor {name} element count {n_elements} is not divisible by cols={cols}"
        ));
    }
    let rows = n_elements / cols;
    load_tensor_quantized(gguf, &name, rows, cols)
}

fn init_weights_from_gguf(
    gguf: &GGUFFile,
    p: &Config,
    debug_mode: bool,
) -> Result<TransformerWeights, String> {
    let head_size = if p.head_dim > 0 {
        p.head_dim
    } else {
        p.dim / p.n_heads
    };
    let kv_dim = p.n_kv_heads * head_size;
    let q_dim = p.n_heads * head_size;
    let n_layers = p.n_layers;
    let ssm_inner = p.ssm_inner_size;
    let ssm_k_heads = p.ssm_group_count;
    let ssm_v_heads = p.ssm_time_step_rank;
    let ssm_head_dim = p.ssm_state_size;
    let ssm_conv_dim = ssm_inner + 2 * ssm_k_heads * ssm_head_dim;

    let token_embedding_table =
        load_tensor_float(gguf, "token_embd.weight", Some(p.vocab_size * p.dim))?;

    let mut rms_att_weight = vec![0.0f32; n_layers * p.dim];
    let mut rms_ffn_weight = vec![0.0f32; n_layers * p.dim];

    let mut wq = vec![QuantizedTensor::default(); n_layers];
    let mut wk = vec![QuantizedTensor::default(); n_layers];
    let mut wv = vec![QuantizedTensor::default(); n_layers];
    let mut wo = vec![QuantizedTensor::default(); n_layers];
    let mut w1 = vec![QuantizedTensor::default(); n_layers];
    let mut w2 = vec![QuantizedTensor::default(); n_layers];
    let mut w3 = vec![QuantizedTensor::default(); n_layers];
    let mut attn_qkv = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_ba = if p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_conv1d = if p.is_qwen3next {
        vec![Vec::new(); n_layers]
    } else {
        Vec::new()
    };
    let mut ssm_a = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_v_heads]
    } else {
        Vec::new()
    };
    let mut ssm_dt_bias = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_v_heads]
    } else {
        Vec::new()
    };
    let mut ssm_norm = if p.is_qwen3next {
        vec![0.0f32; n_layers * ssm_head_dim]
    } else {
        Vec::new()
    };
    let mut moe_gate_inp = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_gate_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_up_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_down_exps = if p.is_qwen3moe || p.is_qwen3next {
        vec![QuantizedTensor::default(); n_layers]
    } else {
        Vec::new()
    };
    let mut moe_shared_gate_inp = if p.is_qwen3next {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };

    let mut attn_q_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * q_dim]
    } else {
        Vec::new()
    };
    let mut attn_k_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * kv_dim]
    } else {
        Vec::new()
    };
    let mut attn_v_bias = if p.is_qwen2 {
        vec![0.0f32; n_layers * kv_dim]
    } else {
        Vec::new()
    };

    let mut attn_q_norm = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![0.0f32; n_layers * head_size]
    } else {
        Vec::new()
    };
    let mut attn_k_norm = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![0.0f32; n_layers * head_size]
    } else {
        Vec::new()
    };
    let mut attn_qk_norm_present = if p.is_gemma3 || p.is_qwen2 || p.is_qwen3moe || p.is_qwen3next {
        vec![false; n_layers]
    } else {
        Vec::new()
    };
    let mut attn_post_norm = if p.is_gemma3 {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };
    let mut ffn_post_norm = if p.is_gemma3 {
        vec![0.0f32; n_layers * p.dim]
    } else {
        Vec::new()
    };

    for l in 0..n_layers {
        let attn_norm = load_layer_tensor_float(gguf, l, "attn_norm.weight", p.dim)?;
        rms_att_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&attn_norm);

        let ffn_norm = if p.is_qwen3next {
            load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p.dim)?
        } else {
            load_layer_tensor_float(gguf, l, "ffn_norm.weight", p.dim)?
        };
        rms_ffn_weight[l * p.dim..(l + 1) * p.dim].copy_from_slice(&ffn_norm);

        if p.is_qwen3next {
            if find_gguf_tensor(gguf, &format!("blk.{l}.attn_qkv.weight")).is_some() {
                attn_qkv[l] =
                    load_layer_tensor_quantized_auto_rows(gguf, l, "attn_qkv.weight", p.dim)?;
                if attn_qkv[l].rows < ssm_conv_dim {
                    return Err(format!(
                        "blk.{l}.attn_qkv.weight has {} rows, expected at least {}",
                        attn_qkv[l].rows,
                        ssm_conv_dim
                    ));
                }
                wo[l] = load_layer_tensor_quantized(gguf, l, "attn_gate.weight", ssm_inner, p.dim)?;
                wv[l] = load_layer_tensor_quantized(gguf, l, "ssm_out.weight", p.dim, ssm_inner)?;
                ssm_ba[l] = load_layer_tensor_quantized(
                    gguf,
                    l,
                    "ssm_ba.weight",
                    2 * ssm_v_heads,
                    p.dim,
                )?;
                ssm_conv1d[l] = load_tensor_float(
                    gguf,
                    &format!("blk.{l}.ssm_conv1d.weight"),
                    Some(p.ssm_conv_kernel * ssm_conv_dim),
                )?;
                let a = load_layer_tensor_float(gguf, l, "ssm_a", ssm_v_heads)?;
                ssm_a[l * ssm_v_heads..(l + 1) * ssm_v_heads].copy_from_slice(&a);
                let dt = load_layer_tensor_float(gguf, l, "ssm_dt.bias", ssm_v_heads)?;
                ssm_dt_bias[l * ssm_v_heads..(l + 1) * ssm_v_heads].copy_from_slice(&dt);
                let n = load_layer_tensor_float(gguf, l, "ssm_norm.weight", ssm_head_dim)?;
                ssm_norm[l * ssm_head_dim..(l + 1) * ssm_head_dim].copy_from_slice(&n);
                if debug_mode {
                    eprintln!(
                        "qwen3next layer {l}: recurrent qkv_rows={} (t={}) gate_rows={} (t={}) out_rows={} (t={})",
                        attn_qkv[l].rows,
                        attn_qkv[l].ttype.0,
                        wo[l].rows,
                        wo[l].ttype.0,
                        wv[l].rows,
                        wv[l].ttype.0
                    );
                    if l == 0 {
                        let (mut amin, mut amax) = (f32::INFINITY, f32::NEG_INFINITY);
                        let (mut dtmin, mut dtmax) = (f32::INFINITY, f32::NEG_INFINITY);
                        for &v in &a {
                            amin = amin.min(v);
                            amax = amax.max(v);
                        }
                        for &v in &dt {
                            dtmin = dtmin.min(v);
                            dtmax = dtmax.max(v);
                        }
                        eprintln!(
                            "qwen3next layer 0: ssm_a[min={amin:.6}, max={amax:.6}] ssm_dt.bias[min={dtmin:.6}, max={dtmax:.6}]"
                        );
                    }
                }
            } else {
                wq[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_q.weight", p.dim)?;
                if wq[l].rows < q_dim {
                    return Err(format!(
                        "blk.{l}.attn_q.weight has {} rows, expected at least {}",
                        wq[l].rows, q_dim
                    ));
                }
                wk[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_k.weight", p.dim)?;
                if wk[l].rows < kv_dim {
                    return Err(format!(
                        "blk.{l}.attn_k.weight has {} rows, expected at least {}",
                        wk[l].rows, kv_dim
                    ));
                }
                wv[l] = load_layer_tensor_quantized_auto_rows(gguf, l, "attn_v.weight", p.dim)?;
                if wv[l].rows < kv_dim {
                    return Err(format!(
                        "blk.{l}.attn_v.weight has {} rows, expected at least {}",
                        wv[l].rows, kv_dim
                    ));
                }
                wo[l] = load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, q_dim)?;
                if debug_mode {
                    eprintln!(
                        "qwen3next layer {l}: full q_rows={} (t={}) k_rows={} (t={}) v_rows={} (t={}) o_rows={} (t={})",
                        wq[l].rows,
                        wq[l].ttype.0,
                        wk[l].rows,
                        wk[l].ttype.0,
                        wv[l].rows,
                        wv[l].ttype.0,
                        wo[l].rows,
                        wo[l].ttype.0
                    );
                }
            }
            moe_gate_inp[l] =
                load_layer_tensor_quantized(gguf, l, "ffn_gate_inp.weight", p.n_experts, p.dim)?;
            moe_gate_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_gate_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_up_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_up_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_down_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_down_exps.weight",
                p.n_experts * p.dim,
                p.expert_hidden_dim,
            )?;

            let shared_hidden = if p.shared_expert_hidden_dim > 0 {
                p.shared_expert_hidden_dim
            } else {
                p.expert_hidden_dim
            };
            w1[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_gate_shexp.weight",
                shared_hidden,
                p.dim,
            )?;
            w2[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_down_shexp.weight",
                p.dim,
                shared_hidden,
            )?;
            w3[l] =
                load_layer_tensor_quantized(gguf, l, "ffn_up_shexp.weight", shared_hidden, p.dim)?;
            let shexp_gate = load_layer_tensor_float(gguf, l, "ffn_gate_inp_shexp.weight", p.dim)?;
            moe_shared_gate_inp[l * p.dim..(l + 1) * p.dim].copy_from_slice(&shexp_gate);
        } else {
            wq[l] = load_layer_tensor_quantized(gguf, l, "attn_q.weight", q_dim, p.dim)?;
            wk[l] = load_layer_tensor_quantized(gguf, l, "attn_k.weight", kv_dim, p.dim)?;
            wv[l] = load_layer_tensor_quantized(gguf, l, "attn_v.weight", kv_dim, p.dim)?;
            wo[l] = load_layer_tensor_quantized(gguf, l, "attn_output.weight", p.dim, q_dim)?;
        }
        if p.is_qwen3moe {
            moe_gate_inp[l] =
                load_layer_tensor_quantized(gguf, l, "ffn_gate_inp.weight", p.n_experts, p.dim)?;
            moe_gate_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_gate_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_up_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_up_exps.weight",
                p.n_experts * p.expert_hidden_dim,
                p.dim,
            )?;
            moe_down_exps[l] = load_layer_tensor_quantized(
                gguf,
                l,
                "ffn_down_exps.weight",
                p.n_experts * p.dim,
                p.expert_hidden_dim,
            )?;
        } else if !p.is_qwen3next {
            w1[l] = load_layer_tensor_quantized(gguf, l, "ffn_gate.weight", p.hidden_dim, p.dim)?;
            w2[l] = load_layer_tensor_quantized(gguf, l, "ffn_down.weight", p.dim, p.hidden_dim)?;
            w3[l] = load_layer_tensor_quantized(gguf, l, "ffn_up.weight", p.hidden_dim, p.dim)?;
        }

        if p.is_qwen2 {
            if let Ok(qb) = load_layer_tensor_float(gguf, l, "attn_q.bias", q_dim) {
                attn_q_bias[l * q_dim..(l + 1) * q_dim].copy_from_slice(&qb);
            }
            if let Ok(kb) = load_layer_tensor_float(gguf, l, "attn_k.bias", kv_dim) {
                attn_k_bias[l * kv_dim..(l + 1) * kv_dim].copy_from_slice(&kb);
            }
            if let Ok(vb) = load_layer_tensor_float(gguf, l, "attn_v.bias", kv_dim) {
                attn_v_bias[l * kv_dim..(l + 1) * kv_dim].copy_from_slice(&vb);
            }
        }

        if p.is_gemma3 || p.is_qwen3moe {
            let q_norm = load_layer_tensor_float(gguf, l, "attn_q_norm.weight", head_size)?;
            let k_norm = load_layer_tensor_float(gguf, l, "attn_k_norm.weight", head_size)?;
            attn_q_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&q_norm);
            attn_k_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&k_norm);
            attn_qk_norm_present[l] = true;
        } else if (p.is_qwen3next || p.is_qwen2)
            && find_gguf_tensor(gguf, &format!("blk.{l}.attn_q_norm.weight")).is_some()
            && find_gguf_tensor(gguf, &format!("blk.{l}.attn_k_norm.weight")).is_some()
        {
            let q_norm = load_layer_tensor_float(gguf, l, "attn_q_norm.weight", head_size)?;
            let k_norm = load_layer_tensor_float(gguf, l, "attn_k_norm.weight", head_size)?;
            attn_q_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&q_norm);
            attn_k_norm[l * head_size..(l + 1) * head_size].copy_from_slice(&k_norm);
            attn_qk_norm_present[l] = true;
        }

        if p.is_gemma3 {
            let pan = load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p.dim)?;
            attn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pan);

            let pfn = load_layer_tensor_float(gguf, l, "post_ffw_norm.weight", p.dim)?;
            ffn_post_norm[l * p.dim..(l + 1) * p.dim].copy_from_slice(&pfn);
        }
    }

    let rms_final_weight = load_tensor_float(gguf, "output_norm.weight", Some(p.dim))?;

    let mut wcls_is_embed = false;
    let wcls = if find_gguf_tensor(gguf, "output.weight").is_some() {
        load_tensor_quantized(gguf, "output.weight", p.vocab_size, p.dim)?
    } else {
        if debug_mode {
            eprintln!("Using tied embeddings for output projection");
        }
        wcls_is_embed = true;
        QuantizedTensor {
            data_offset: usize::MAX,
            ttype: GgmlType(GGML_TYPE_F32),
            rows: p.vocab_size,
            cols: p.dim,
        }
    };

    Ok(TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        attn_qkv,
        ssm_ba,
        ssm_conv1d,
        ssm_a,
        ssm_dt_bias,
        ssm_norm,
        moe_gate_inp,
        moe_gate_exps,
        moe_up_exps,
        moe_down_exps,
        moe_shared_gate_inp,
        rms_final_weight,
        wcls,
        wcls_is_embed,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
        attn_q_norm,
        attn_k_norm,
        attn_qk_norm_present,
        attn_post_norm,
        ffn_post_norm,
    })
}

fn malloc_run_state(p: &Config) -> RunState {
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
    let scratch_dim = ffn_dim
        .max(ssm_conv_dim)
        .max(ssm_inner)
        .max(ssm_head_dim);

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

fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32) {
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

fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], size: usize, eps: f32) {
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

fn rmsnorm_gemma(o: &mut [f32], x: &[f32], weight: &[f32], size: usize, eps: f32) {
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

fn rmsnorm_per_head_gemma_inplace(
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

fn softmax(x: &mut [f32], size: usize) {
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
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn siluf(x: f32) -> f32 {
    x * sigmoidf(x)
}

#[inline(always)]
fn softplusf(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

#[inline(always)]
fn l2_norm(x: &[f32]) -> f32 {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss.sqrt()
}

#[inline(always)]
fn qwen3next_state_head_step(
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

fn qwen3next_linear_attention_autoregressive(
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
    if n_v_heads % n_k_heads != 0 {
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
                qwen3next_state_head_step(state_h, out_h, kv_mem, delta, q, k, v, gate_all[h], beta_all[h]);
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
            qwen3next_state_head_step(state_h, out_h, kv_mem, delta, q, k, v, gate_all[h], beta_all[h]);
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

fn matmul_f32_embeddings(logits: &mut [f32], x: &[f32], emb: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &emb[r * cols..(r + 1) * cols];
        logits[r] = dot_f32_simd(row, &x[..cols]);
    }
}

fn transformer(
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
    let layer_debug = env::var("LLAMA3PURE_LAYER_DEBUG")
        .ok()
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let layer_debug_pos = env::var("LLAMA3PURE_LAYER_DEBUG_POS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let do_layer_debug = layer_debug && layer_debug_pos.map_or(pos == 0, |p0| pos == p0);

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
                        s.q[..q_dim]
                            .par_chunks_mut(head_size)
                            .enumerate()
                            .for_each(|(h, q_dst)| {
                                let src_base = h * 2 * head_size;
                                q_dst.copy_from_slice(&hb_src[src_base..src_base + head_size]);
                            });
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

fn tiktoken_decode_map() -> [i16; 512] {
    let mut map = [-1i16; 512];
    let mut n = 0i16;
    for b in 0..=255u16 {
        let b8 = b as u8;
        if (33..=126).contains(&b8) || (161..=172).contains(&b8) || (174..=255).contains(&b8) {
            map[b as usize] = b as i16;
        } else {
            map[(256 + n as u16) as usize] = b as i16;
            n += 1;
        }
    }
    map
}

fn tiktoken_encode_map() -> [u32; 256] {
    let mut map = [0u32; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let b8 = b as u8;
        if (33..=126).contains(&b8) || (161..=172).contains(&b8) || (174..=255).contains(&b8) {
            map[b as usize] = b;
        } else {
            map[b as usize] = 256 + n;
            n += 1;
        }
    }
    map
}

fn decode_sentencepiece(s: &str) -> String {
    s.replace('\u{2581}', " ")
}

fn decode_tiktoken_internal(s: &str) -> String {
    let map = tiktoken_decode_map();
    let mut out: Vec<u8> = Vec::with_capacity(s.len());

    for ch in s.chars() {
        let cp = ch as u32;
        if cp < 512 {
            let v = map[cp as usize];
            if v >= 0 {
                out.push(v as u8);
                continue;
            }
        }
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        out.extend_from_slice(encoded.as_bytes());
    }

    String::from_utf8_lossy(&out).to_string()
}

fn text_to_tiktoken(text: &str) -> String {
    let map = tiktoken_encode_map();
    let mut out = String::with_capacity(text.len() * 2);
    for b in text.as_bytes() {
        let cp = map[*b as usize];
        if let Some(ch) = char::from_u32(cp) {
            out.push(ch);
        }
    }
    out
}

fn text_to_sentencepiece(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    let mut need_prefix = true;

    for b in text.bytes() {
        match b {
            b' ' => {
                out.push('\u{2581}');
                need_prefix = false;
            }
            b'\n' | b'\t' | b'\r' => {
                out.push(b as char);
                need_prefix = true;
            }
            _ => {
                if need_prefix && (b as char).is_ascii_alphanumeric() {
                    out.push('\u{2581}');
                }
                out.push(b as char);
                need_prefix = false;
            }
        }
    }

    out
}

fn split_gpt2_pieces(text: &str) -> Vec<String> {
    fn contraction_len(s: &str, idx: usize) -> usize {
        let rest = &s[idx..];
        for pat in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
            if rest.starts_with(pat) {
                return pat.len();
            }
        }
        0
    }

    fn next_char(s: &str, idx: usize) -> Option<(char, usize)> {
        s[idx..].chars().next().map(|c| (c, c.len_utf8()))
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    enum Kind {
        Alpha,
        Numeric,
        Other,
    }

    fn char_kind(c: char) -> Kind {
        if c.is_alphabetic() {
            Kind::Alpha
        } else if c.is_numeric() {
            Kind::Numeric
        } else {
            Kind::Other
        }
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    let len = text.len();

    while i < len {
        let (c0, c0_len) = match next_char(text, i) {
            Some(v) => v,
            None => break,
        };

        if c0.is_whitespace() && c0 != ' ' {
            let start = i;
            i += c0_len;
            while i < len {
                if let Some((c, clen)) = next_char(text, i) {
                    if c.is_whitespace() && c != ' ' {
                        i += clen;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            out.push(text[start..i].to_string());
            continue;
        }

        if c0 == ' ' {
            let mut j = i + c0_len;
            if j >= len {
                out.push(" ".to_string());
                break;
            }
            if let Some((c1, _)) = next_char(text, j) {
                if c1.is_whitespace() {
                    let start = i;
                    while j < len {
                        if let Some((c, clen)) = next_char(text, j) {
                            if c.is_whitespace() {
                                j += clen;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    out.push(text[start..j].to_string());
                    i = j;
                    continue;
                }
            }

            let start = i;
            i = j;
            let contr = contraction_len(text, i);
            if contr > 0 {
                i += contr;
                out.push(text[start..i].to_string());
                continue;
            }
            if let Some((c1, clen1)) = next_char(text, i) {
                let kind = char_kind(c1);
                i += clen1;
                while i < len {
                    let contr2 = contraction_len(text, i);
                    if contr2 > 0 {
                        break;
                    }
                    if let Some((c, clen)) = next_char(text, i) {
                        if c.is_whitespace() {
                            break;
                        }
                        if char_kind(c) != kind {
                            break;
                        }
                        i += clen;
                    } else {
                        break;
                    }
                }
                out.push(text[start..i].to_string());
                continue;
            }
        }

        let contr = contraction_len(text, i);
        if contr > 0 {
            let start = i;
            i += contr;
            out.push(text[start..i].to_string());
            continue;
        }

        let start = i;
        let kind = char_kind(c0);
        i += c0_len;
        while i < len {
            let contr2 = contraction_len(text, i);
            if contr2 > 0 {
                break;
            }
            if let Some((c, clen)) = next_char(text, i) {
                if c.is_whitespace() {
                    break;
                }
                if char_kind(c) != kind {
                    break;
                }
                i += clen;
            } else {
                break;
            }
        }
        out.push(text[start..i].to_string());
    }

    out
}

impl Tokenizer {
    fn find_special_token(&self, token_str: &str) -> Option<i32> {
        self.vocab
            .iter()
            .position(|s| s == token_str)
            .map(|i| i as i32)
    }

    fn build_token_lookup(&mut self) {
        if !self.token_to_id.is_empty() {
            return;
        }
        let mut map = HashMap::with_capacity(self.vocab.len() * 2);
        for (id, tok) in self.vocab.iter().enumerate() {
            map.entry(tok.clone()).or_insert(id as i32);
        }
        self.token_to_id = map;
    }

    fn build_merge_ranks(&mut self) {
        if !self.merge_ranks.is_empty() {
            return;
        }
        let mut ranks = HashMap::with_capacity(self.merges.len() * 2);
        for (rank, m) in self.merges.iter().enumerate() {
            ranks.entry(m.clone()).or_insert(rank);
        }
        self.merge_ranks = ranks;
    }

    fn bpe_encode(&mut self, text: &str, tokens: &mut Vec<i32>) {
        tokens.clear();
        if text.is_empty() {
            return;
        }

        if self.use_sentencepiece {
            let encoded_text = text_to_sentencepiece(text);
            self.build_token_lookup();

            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                return;
            }

            while work.len() > 1 {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_id = -1i32;
                let mut best_pos = 0usize;

                for i in 0..work.len() - 1 {
                    let left = &self.vocab[work[i] as usize];
                    let right = &self.vocab[work[i + 1] as usize];
                    let merged = format!("{left}{right}");
                    if let Some(&id) = self.token_to_id.get(&merged) {
                        let score = self.vocab_scores.get(id as usize).copied().unwrap_or(0.0);
                        if score > best_score {
                            best_score = score;
                            best_id = id;
                            best_pos = i;
                        }
                    }
                }

                if best_id < 0 {
                    break;
                }

                work[best_pos] = best_id;
                work.remove(best_pos + 1);
            }

            tokens.extend(work);
            return;
        }

        self.build_token_lookup();
        self.build_merge_ranks();

        let pieces = split_gpt2_pieces(text);
        for piece in pieces {
            let encoded_text = text_to_tiktoken(&piece);
            let mut work: Vec<i32> = Vec::with_capacity(encoded_text.len());
            for ch in encoded_text.chars() {
                let s = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&s) {
                    work.push(id);
                }
            }
            if work.is_empty() {
                continue;
            }

            while work.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_id = -1i32;
                let mut best_pos = 0usize;

                for i in 0..work.len() - 1 {
                    let left = &self.vocab[work[i] as usize];
                    let right = &self.vocab[work[i + 1] as usize];
                    let pair = format!("{left} {right}");
                    let merged = format!("{left}{right}");
                    if let Some(&rank) = self.merge_ranks.get(&pair) {
                        if let Some(&id) = self.token_to_id.get(&merged) {
                            if rank < best_rank {
                                best_rank = rank;
                                best_id = id;
                                best_pos = i;
                            }
                        }
                    }
                }

                if best_id < 0 {
                    break;
                }

                work[best_pos] = best_id;
                work.remove(best_pos + 1);
            }

            tokens.extend(work);
        }

        return;
    }

    fn decode_token(&self, token_id: i32) -> Option<String> {
        if token_id < 0 || token_id as usize >= self.vocab.len() {
            return None;
        }
        let raw = &self.vocab[token_id as usize];
        if self.use_sentencepiece {
            Some(decode_sentencepiece(raw))
        } else {
            Some(decode_tiktoken_internal(raw))
        }
    }
}

fn init_tokenizer_from_gguf(
    gguf: &GGUFFile,
    config: &mut Config,
    debug_mode: bool,
) -> Result<Tokenizer, String> {
    if gguf.vocab_tokens.is_empty() {
        return Err("no vocabulary found in GGUF file".to_string());
    }

    let mut tokenizer = Tokenizer::default();
    tokenizer.bos_token = match gguf.kv.get("tokenizer.ggml.bos_token_id") {
        Some(GgufValue::UInt(v)) => *v as i32,
        Some(GgufValue::Int(v)) => *v as i32,
        _ => -1,
    };
    tokenizer.eos_token = get_gguf_int_from_map(
        &gguf.kv,
        "tokenizer.ggml.eos_token_id",
        LLAMA3_EOS_TOKEN as i64,
    ) as i32;
    tokenizer.start_header_token = LLAMA3_START_HEADER;
    tokenizer.end_header_token = LLAMA3_END_HEADER;
    tokenizer.eot_token = LLAMA3_EOT;

    tokenizer.vocab = gguf.vocab_tokens.clone();
    tokenizer.vocab_size = tokenizer.vocab.len();
    tokenizer.max_token_length = tokenizer
        .vocab
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(256)
        .max(1);
    tokenizer.vocab_scores = if gguf.vocab_scores.is_empty() {
        vec![0.0; tokenizer.vocab_size]
    } else {
        gguf.vocab_scores.clone()
    };
    tokenizer.merges = gguf.vocab_merges.clone();
    if tokenizer.bos_token < 0 {
        if config.is_qwen2 || config.is_qwen3moe || config.is_qwen3next {
            tokenizer.bos_token = -1;
        } else {
            tokenizer.bos_token = tokenizer
                .vocab
                .iter()
                .position(|s| s == "<|begin_of_text|>")
                .map(|i| i as i32)
                .or_else(|| tokenizer.vocab.iter().position(|s| s == "<s>").map(|i| i as i32))
                .unwrap_or(LLAMA3_BOS_TOKEN);
        }
    }

    if debug_mode {
        eprintln!(
            "Using vocabulary from GGUF file ({} tokens)",
            tokenizer.vocab_size
        );
    }

    if config.vocab_size != tokenizer.vocab_size {
        if debug_mode {
            eprintln!(
                "Note: Updating vocab_size from {} to {} based on GGUF",
                config.vocab_size, tokenizer.vocab_size
            );
        }
        config.vocab_size = tokenizer.vocab_size;
    }

    Ok(tokenizer)
}

fn argmax(v: &[f32]) -> usize {
    let mut max_i = 0usize;
    let mut max_p = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_p {
            max_p = val;
            max_i = i;
        }
    }
    max_i
}

fn sample(probabilities: &[f32], rng: &mut XorShiftRng) -> usize {
    let r = rng.random_f32();
    let mut cdf = 0.0f32;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probabilities.len().saturating_sub(1)
}

#[derive(Clone, Copy)]
struct Candidate {
    idx: usize,
    score: f32,
}

struct TopKSampler {
    candidates: Vec<Candidate>,
    probs: Vec<f32>,
}

impl TopKSampler {
    fn new() -> Self {
        Self {
            candidates: Vec::new(),
            probs: Vec::new(),
        }
    }

    fn find_min_pos(cands: &[Candidate]) -> usize {
        let mut min_pos = 0usize;
        let mut min_score = cands[0].score;
        for (i, c) in cands.iter().enumerate().skip(1) {
            if c.score < min_score {
                min_score = c.score;
                min_pos = i;
            }
        }
        min_pos
    }

    fn sample_top_k_top_p(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut XorShiftRng,
    ) -> usize {
        let k = top_k.min(logits.len()).max(1);
        self.candidates.clear();
        if self.candidates.capacity() < k {
            self.candidates.reserve(k - self.candidates.capacity());
        }

        let mut min_pos = 0usize;
        for (idx, &logit) in logits.iter().enumerate() {
            let score = logit / temperature;
            if self.candidates.len() < k {
                self.candidates.push(Candidate { idx, score });
                min_pos = Self::find_min_pos(&self.candidates);
            } else if score > self.candidates[min_pos].score {
                self.candidates[min_pos] = Candidate { idx, score };
                min_pos = Self::find_min_pos(&self.candidates);
            }
        }

        self.candidates
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

        let max_score = self.candidates[0].score;
        self.probs.clear();
        if self.probs.capacity() < self.candidates.len() {
            self.probs
                .reserve(self.candidates.len() - self.probs.capacity());
        }

        let mut prob_sum = 0.0f32;
        for c in &self.candidates {
            let p = (c.score - max_score).exp();
            self.probs.push(p);
            prob_sum += p;
        }

        let mut keep = self.candidates.len();
        if top_p < 1.0 {
            let mut cumulative = 0.0f32;
            keep = 0;
            for &p in &self.probs {
                cumulative += p / prob_sum;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
        }

        let kept_sum: f32 = self.probs[..keep].iter().copied().sum();
        let mut r = rng.random_f32() * kept_sum;
        for i in 0..keep {
            r -= self.probs[i];
            if r <= 0.0 {
                return self.candidates[i].idx;
            }
        }
        self.candidates[keep - 1].idx
    }
}

fn time_in_ms() -> i64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() * 1000 + (now.subsec_nanos() as u64 / 1_000_000)) as i64
}

fn configure_rayon_threads(num_threads: usize, debug_mode: bool) {
    if num_threads == 0 {
        return;
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(()) => {
            if debug_mode {
                eprintln!("Parallel: configured rayon worker threads={num_threads}");
            }
        }
        Err(e) => {
            if debug_mode {
                eprintln!(
                    "Parallel: keeping existing rayon global thread pool (requested {num_threads}, reason: {e})"
                );
            }
        }
    }
}

fn usage(program: &str) {
    println!("Usage: {program} -model <model.gguf> -prompt <text> [-url <remote.gguf>] [options]");
    println!();
    println!("Required arguments:");
    println!("  -model         path to GGUF model file");
    println!("  -prompt        input prompt text");
    println!();
    println!("Optional arguments:");
    println!("  -url           remote GGUF URL (used only when local -model file is missing/invalid)");
    println!("  -system_prompt system prompt (default: \"You are a helpful assistant.\")");
    println!("  -temperature   sampling temperature (default: 0.9, use 0.0 for greedy)");
    println!("  -top_k         top-k sampling cutoff (default: 0 = disabled)");
    println!("  -top_p         top-p nucleus threshold in (0,1] (default: 1.0 = disabled)");
    println!("  -max_tokens    number of tokens to generate (default: 256)");
    println!("  -context_size  context size for the AI model (default: model's max)");
    println!("  -threads       rayon worker threads (default: auto)");
    println!("  -show-tokens   always print achieved tok/s at the end");
    println!("  -profiling     print token-level profiling counters");
    println!("  -debug         show detailed model loading and performance logs");
    println!();
    println!("Example:");
    println!("  {program} -model Llama3.gguf -prompt \"tell me what is microsoft\"");
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut temperature: f32 = 0.9;
    let mut top_k: usize = 0;
    let mut top_p: f32 = 1.0;
    let mut max_tokens: usize = 256;
    let mut context_size: usize = 0;
    let mut rayon_threads: Option<usize> = env::var("LLAMA3PURE_RAYON_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let mut system_prompt = String::from("You are a helpful assistant.");
    let mut checkpoint: Option<String> = None;
    let mut model_url: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut profiling_mode = false;
    let mut show_tokens = false;
    let mut debug_mode = false;

    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| "llama3pure".to_string());

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-model" if i + 1 < args.len() => {
                checkpoint = Some(args[i + 1].clone());
                i += 2;
            }
            "-url" if i + 1 < args.len() => {
                model_url = Some(args[i + 1].clone());
                i += 2;
            }
            "-temperature" if i + 1 < args.len() => {
                temperature = args[i + 1]
                    .parse::<f32>()
                    .map_err(|e| format!("invalid -temperature: {e}"))?;
                i += 2;
            }
            "-top_k" if i + 1 < args.len() => {
                top_k = args[i + 1]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid -top_k: {e}"))?;
                i += 2;
            }
            "-top_p" if i + 1 < args.len() => {
                top_p = args[i + 1]
                    .parse::<f32>()
                    .map_err(|e| format!("invalid -top_p: {e}"))?;
                if !(top_p > 0.0 && top_p <= 1.0) {
                    return Err("invalid -top_p: expected value in (0, 1]".to_string());
                }
                i += 2;
            }
            "-max_tokens" if i + 1 < args.len() => {
                max_tokens = args[i + 1]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid -max_tokens: {e}"))?;
                i += 2;
            }
            "-context_size" if i + 1 < args.len() => {
                context_size = args[i + 1]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid -context_size: {e}"))?;
                i += 2;
            }
            "-threads" if i + 1 < args.len() => {
                rayon_threads = Some(
                    args[i + 1]
                        .parse::<usize>()
                        .map_err(|e| format!("invalid -threads: {e}"))?,
                );
                if rayon_threads == Some(0) {
                    return Err("invalid -threads: expected >= 1".to_string());
                }
                i += 2;
            }
            "-profiling" => {
                profiling_mode = true;
                i += 1;
            }
            "-show-tokens" | "-show_tokens" => {
                show_tokens = true;
                i += 1;
            }
            "-prompt" if i + 1 < args.len() => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "-system_prompt" if i + 1 < args.len() => {
                system_prompt = args[i + 1].clone();
                i += 2;
            }
            "-debug" => {
                debug_mode = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let checkpoint = checkpoint.ok_or_else(|| {
        usage(&program);
        "missing required -model".to_string()
    })?;
    let prompt = prompt.ok_or_else(|| {
        usage(&program);
        "missing required -prompt".to_string()
    })?;

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

    let mut config = model::config::build_config_from_gguf(&gguf, debug_mode)?;

    let mut tokenizer = init_tokenizer_from_gguf(&gguf, &mut config, debug_mode)?;
    tokenizer.use_sentencepiece = config.is_gemma3;

    model::config::apply_context_size_overrides(&mut config, context_size, debug_mode);
    if max_tokens == 0 || max_tokens > config.seq_len {
        max_tokens = config.seq_len;
    }

    if let Some(n_threads) = rayon_threads {
        configure_rayon_threads(n_threads, debug_mode);
    }

    PROFILING_ENABLED.store(profiling_mode, AtomicOrdering::Relaxed);
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

    let weights = init_weights_from_gguf(&gguf, &config, debug_mode)?;

    let mut state = malloc_run_state(&config);

    let use_chat_template = true;

    let mut prompt_tokens: Vec<i32> = if use_chat_template {
        model::chat::encode_chat_prompt(&mut tokenizer, &config, &prompt, &system_prompt)
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
        transformer(
            token as usize,
            pos,
            &config,
            &mut state,
            &weights,
            gguf.mapped.as_slice(),
        )?;
        prof_end(&PROF_TRANSFORMER_NS, prof_t0);
        if profiling_mode {
            PROF_FORWARD_PASSES.fetch_add(1, AtomicOrdering::Relaxed);
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
        let total_ns = PROF_TRANSFORMER_NS.load(AtomicOrdering::Relaxed);
        let matmul_ns = PROF_MATMUL_NS.load(AtomicOrdering::Relaxed);
        let ssm_ns = PROF_SSM_NS.load(AtomicOrdering::Relaxed);
        let attn_ns = PROF_ATTN_NS.load(AtomicOrdering::Relaxed);
        let moe_ns = PROF_MOE_NS.load(AtomicOrdering::Relaxed);
        let ffn_ns = PROF_FFN_NS.load(AtomicOrdering::Relaxed);
        let passes = PROF_FORWARD_PASSES.load(AtomicOrdering::Relaxed);
        let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
        let pct = |part: u64| {
            if total_ns == 0 {
                0.0
            } else {
                (part as f64 * 100.0) / total_ns as f64
            }
        };
        eprintln!("\n[PROFILE] forward_passes={passes}");
        eprintln!(
            "[PROFILE] transformer_total={:.3} ms ({:.3} ms/pass)",
            to_ms(total_ns),
            if passes == 0 { 0.0 } else { to_ms(total_ns) / passes as f64 }
        );
        eprintln!(
            "[PROFILE] matmul={:.3} ms ({:.1}%)",
            to_ms(matmul_ns),
            pct(matmul_ns)
        );
        eprintln!("[PROFILE] ssm={:.3} ms ({:.1}%)", to_ms(ssm_ns), pct(ssm_ns));
        eprintln!(
            "[PROFILE] attention={:.3} ms ({:.1}%)",
            to_ms(attn_ns),
            pct(attn_ns)
        );
        eprintln!("[PROFILE] moe={:.3} ms ({:.1}%)", to_ms(moe_ns), pct(moe_ns));
        eprintln!("[PROFILE] ffn={:.3} ms ({:.1}%)", to_ms(ffn_ns), pct(ffn_ns));
        eprintln!("[PROFILE] note: counters overlap (e.g. matmul is included in SSM/attention/MoE/FFN)");
    }

    Ok(())
}
