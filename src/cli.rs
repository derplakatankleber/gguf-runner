use clap::Parser;

fn parse_top_p(raw: &str) -> Result<f32, String> {
    let v = raw
        .parse::<f32>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected value in (0, 1]"))
    }
}

fn parse_positive_usize(raw: &str) -> Result<usize, String> {
    let v = raw
        .parse::<usize>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected >= 1"))
    }
}

fn parse_boolish(raw: &str) -> Result<bool, String> {
    let v = raw.trim();
    if v.eq_ignore_ascii_case("1")
        || v.eq_ignore_ascii_case("true")
        || v.eq_ignore_ascii_case("yes")
        || v.eq_ignore_ascii_case("on")
    {
        return Ok(true);
    }
    if v.eq_ignore_ascii_case("0")
        || v.eq_ignore_ascii_case("false")
        || v.eq_ignore_ascii_case("no")
        || v.eq_ignore_ascii_case("off")
    {
        return Ok(false);
    }
    Err(format!(
        "invalid value '{raw}': expected one of 1/0/true/false/yes/no/on/off"
    ))
}

#[derive(Parser, Debug)]
#[command(
    about = "Run GGUF language models",
    long_about = None,
    disable_help_subcommand = true
)]
struct Cli {
    #[arg(long, required = true, value_name = "model.gguf")]
    model: String,

    #[arg(long, required = true)]
    prompt: String,

    #[arg(long)]
    url: Option<String>,

    #[arg(long, default_value_t = 0.9)]
    temperature: f32,

    #[arg(long = "top-k", default_value_t = 0)]
    top_k: usize,

    #[arg(
        long = "top-p",
        value_parser = parse_top_p,
        default_value_t = 1.0
    )]
    top_p: f32,

    #[arg(long = "max-tokens", default_value_t = 256)]
    max_tokens: usize,

    #[arg(long = "context-size", default_value_t = 0)]
    context_size: usize,

    #[arg(
        long,
        env = "GGUF_RAYON_THREADS",
        value_parser = parse_positive_usize
    )]
    threads: Option<usize>,

    #[arg(
        long = "system-prompt",
        default_value = "You are a helpful assistant."
    )]
    system_prompt: String,

    #[arg(long)]
    profiling: bool,

    #[arg(long = "show-tokens")]
    show_tokens: bool,

    #[arg(long = "show-timings")]
    show_timings: bool,

    #[arg(long)]
    debug: bool,

    #[arg(
        long = "par-matmul-min-rows",
        hide = true,
        env = "GGUF_PAR_MATMUL_MIN_ROWS",
        value_parser = parse_positive_usize
    )]
    par_matmul_min_rows: Option<usize>,

    #[arg(
        long = "par-matmul-chunk-rows",
        hide = true,
        env = "GGUF_PAR_MATMUL_CHUNK_ROWS",
        value_parser = parse_positive_usize
    )]
    par_matmul_chunk_rows: Option<usize>,

    #[arg(
        long = "par-attn-min-heads",
        hide = true,
        env = "GGUF_PAR_ATTN_MIN_HEADS",
        value_parser = parse_positive_usize
    )]
    par_attn_min_heads: Option<usize>,

    #[arg(
        long = "par-qwen3next-min-heads",
        hide = true,
        env = "GGUF_PAR_QWEN3NEXT_MIN_HEADS",
        value_parser = parse_positive_usize
    )]
    par_qwen3next_min_heads: Option<usize>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-dotprod-q8",
        hide = true,
        env = "GGUF_AARCH64_DOTPROD_Q8",
        value_parser = parse_boolish
    )]
    aarch64_dotprod_q8: Option<bool>,

    #[cfg(target_arch = "aarch64")]
    #[arg(
        long = "aarch64-qk-mr4",
        hide = true,
        env = "GGUF_AARCH64_QK_MR4",
        value_parser = parse_boolish
    )]
    aarch64_qk_mr4: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avx2",
        hide = true,
        env = "GGUF_X86_AVX2",
        value_parser = parse_boolish
    )]
    x86_avx2: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-f16c",
        hide = true,
        env = "GGUF_X86_F16C",
        value_parser = parse_boolish
    )]
    x86_f16c: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-qk-mr4",
        hide = true,
        env = "GGUF_X86_QK_MR4",
        value_parser = parse_boolish
    )]
    x86_qk_mr4: Option<bool>,

    #[arg(
        long = "layer-debug",
        hide = true,
        env = "GGUF_LAYER_DEBUG",
        value_parser = parse_boolish
    )]
    layer_debug: Option<bool>,

    #[arg(
        long = "layer-debug-pos",
        hide = true,
        env = "GGUF_LAYER_DEBUG_POS"
    )]
    layer_debug_pos: Option<usize>,
}

pub(crate) struct CliOptions {
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) url: Option<String>,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) max_tokens: usize,
    pub(crate) context_size: usize,
    pub(crate) threads: Option<usize>,
    pub(crate) system_prompt: String,
    pub(crate) profiling: bool,
    pub(crate) show_tokens: bool,
    pub(crate) show_timings: bool,
    pub(crate) debug: bool,
    pub(crate) par_matmul_min_rows: Option<usize>,
    pub(crate) par_matmul_chunk_rows: Option<usize>,
    pub(crate) par_attn_min_heads: Option<usize>,
    pub(crate) par_qwen3next_min_heads: Option<usize>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_dotprod_q8: Option<bool>,
    #[cfg(target_arch = "aarch64")]
    pub(crate) aarch64_qk_mr4: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx2: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_f16c: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_qk_mr4: Option<bool>,
    pub(crate) layer_debug: Option<bool>,
    pub(crate) layer_debug_pos: Option<usize>,
}

impl CliOptions {
    pub(crate) fn parse() -> Result<Self, String> {
        let cli = Cli::try_parse()
            .map_err(|e| e.to_string())?;

        Ok(Self {
            model: cli.model,
            prompt: cli.prompt,
            url: cli.url,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            max_tokens: cli.max_tokens,
            context_size: cli.context_size,
            threads: cli.threads,
            system_prompt: cli.system_prompt,
            profiling: cli.profiling,
            show_tokens: cli.show_tokens,
            show_timings: cli.show_timings,
            debug: cli.debug,
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
        })
    }
}
