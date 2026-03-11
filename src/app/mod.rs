mod agent;
mod generation;

use crate::cli::CliOperationMode;
use crate::cli::CliOptions;
use crate::engine::profiling::{print_profile_report, profiling_reset, set_profiling_enabled};
#[cfg(target_arch = "aarch64")]
use crate::engine::switches::aarch64_matmul_prefetch_rows;
use crate::engine::switches::{
    init_runtime_config, kv_cache_mode, par_attn_min_heads, par_matmul_chunk_rows,
    par_matmul_min_rows, par_qwen3next_min_heads, KvCacheMode, RuntimeSwitchConfig,
};
use crate::engine::types::{ContentPart, GenerationRequest, MediaRef};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_IMAGES: usize = 10;
const MAX_VIDEOS: usize = 10;
const MAX_AUDIOS: usize = 10;

const MAX_IMAGE_BYTES: u64 = 50 * 1024 * 1024;
const MAX_VIDEO_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_AUDIO_BYTES: u64 = 1024 * 1024 * 1024;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4"];
const REPL_COMMANDS: [&str; 4] = ["help", "model", "exit", "quit"];

fn map_kv_cache_mode(mode: Option<crate::cli::CliKvCacheMode>) -> Option<KvCacheMode> {
    mode.map(|v| match v {
        crate::cli::CliKvCacheMode::Auto => KvCacheMode::Auto,
        crate::cli::CliKvCacheMode::Q8 => KvCacheMode::Q8,
        crate::cli::CliKvCacheMode::Q4 => KvCacheMode::Q4,
    })
}

fn print_cpu_features() {
    fn yn(v: bool) -> &'static str {
        if v {
            "yes"
        } else {
            "no "
        }
    }

    println!("Architecture: {}", std::env::consts::ARCH);
    println!();

    #[cfg(target_arch = "aarch64")]
    {
        let features: &[(&str, &str, bool)] = &[
            (
                "neon",
                "ARMv8-A (baseline)",
                std::arch::is_aarch64_feature_detected!("neon"),
            ),
            (
                "dotprod",
                "ARMv8.2-A",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            (
                "fp16",
                "ARMv8.2-A",
                std::arch::is_aarch64_feature_detected!("fp16"),
            ),
            (
                "i8mm",
                "ARMv8.6-A",
                std::arch::is_aarch64_feature_detected!("i8mm"),
            ),
            (
                "sve",
                "ARMv8.4-A (opt-in)",
                std::arch::is_aarch64_feature_detected!("sve"),
            ),
            (
                "sve2",
                "ARMv9-A",
                std::arch::is_aarch64_feature_detected!("sve2"),
            ),
        ];
        println!("{:<10}  {:<20}  {:>8}", "feature", "ISA", "runtime");
        println!("{}", "-".repeat(44));
        for (name, isa, runtime) in features {
            println!("{:<10}  {:<20}  {:>8}", name, isa, yn(*runtime));
        }
        println!();
        println!("gguf-runner kernels (aarch64):");
        println!("  NEON matmul Q4/Q5/Q6-K MR4:  always enabled");
        println!("  FCVTL fp16 loads:             always enabled (base AArch64)");
        println!("  VSHLL bf16 loads:             always enabled (base AArch64)");
        println!(
            "  dotprod Q8_0:                 runtime={}",
            yn(std::arch::is_aarch64_feature_detected!("dotprod"))
        );
        println!(
            "  i8mm Q8_0 MR2 (SMMLA):       runtime={}",
            yn(std::arch::is_aarch64_feature_detected!("i8mm"))
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        let features: &[(&str, &str, bool)] = &[
            (
                "sse4.1",
                "Intel Penryn 2007",
                std::arch::is_x86_feature_detected!("sse4.1"),
            ),
            (
                "avx",
                "Intel Sandy Br. 2011",
                std::arch::is_x86_feature_detected!("avx"),
            ),
            (
                "avx2",
                "Intel Haswell 2013",
                std::arch::is_x86_feature_detected!("avx2"),
            ),
            (
                "fma",
                "Intel Haswell 2013",
                std::arch::is_x86_feature_detected!("fma"),
            ),
            (
                "f16c",
                "Intel Ivy Br. 2012",
                std::arch::is_x86_feature_detected!("f16c"),
            ),
            (
                "avxvnni",
                "Intel Alder Lk. 2021",
                std::arch::is_x86_feature_detected!("avxvnni"),
            ),
            (
                "avx512f",
                "Intel Skylake-X 2017",
                std::arch::is_x86_feature_detected!("avx512f"),
            ),
            (
                "avx512vnni",
                "Intel Cascade Lk. 2019",
                std::arch::is_x86_feature_detected!("avx512vnni"),
            ),
            (
                "avx512vl",
                "Intel Skylake-X 2017",
                std::arch::is_x86_feature_detected!("avx512vl"),
            ),
        ];
        println!("{:<12}  {:<24}  {:>8}", "feature", "ISA", "runtime");
        println!("{}", "-".repeat(50));
        for (name, isa, runtime) in features {
            println!("{:<12}  {:<24}  {:>8}", name, isa, yn(*runtime));
        }
        println!();
        println!("gguf-runner kernels (x86_64):");
        println!(
            "  AVX2+FMA matmul Q4/Q5/Q6-K:  runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
        );
        println!(
            "  F16C fp16 loads:              runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c"))
        );
        println!(
            "  AVX-VNNI Q8_0:                runtime={}",
            yn(std::arch::is_x86_feature_detected!("avxvnni"))
        );
        println!(
            "  AVX-512VNNI Q8_0:             runtime={}",
            yn(std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl"))
        );
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    println!("No architecture-specific features detected for this target.");
}

pub(crate) fn run() -> Result<(), String> {
    let cli = CliOptions::parse()?;

    if cli.show_features {
        print_cpu_features();
        return Ok(());
    }

    let runtime_switch_config = RuntimeSwitchConfig {
        par_matmul_min_rows: cli.par_matmul_min_rows,
        par_matmul_chunk_rows: cli.par_matmul_chunk_rows,
        #[cfg(target_arch = "aarch64")]
        aarch64_matmul_prefetch_rows: cli.aarch64_matmul_prefetch_rows,
        par_attn_min_heads: cli.par_attn_min_heads,
        par_qwen3next_min_heads: cli.par_qwen3next_min_heads,
        #[cfg(target_arch = "aarch64")]
        aarch64_dotprod_q8: cli.aarch64_dotprod_q8,
        #[cfg(target_arch = "aarch64")]
        aarch64_qk_mr4: cli.aarch64_qk_mr4,
        #[cfg(target_arch = "aarch64")]
        aarch64_i8mm: cli.aarch64_i8mm,
        #[cfg(target_arch = "x86_64")]
        x86_avx2: cli.x86_avx2,
        #[cfg(target_arch = "x86_64")]
        x86_f16c: cli.x86_f16c,
        #[cfg(target_arch = "x86_64")]
        x86_qk_mr4: cli.x86_qk_mr4,
        #[cfg(target_arch = "x86_64")]
        x86_avxvnni: cli.x86_avxvnni,
        #[cfg(target_arch = "x86_64")]
        x86_avx512vnni_q8: cli.x86_avx512vnni_q8,
        layer_debug: cli.layer_debug,
        layer_debug_pos: cli.layer_debug_pos,
        kv_cache_mode: map_kv_cache_mode(cli.kv_cache_mode),
    };
    init_runtime_config(&runtime_switch_config);
    let run_started = Instant::now();

    set_profiling_enabled(cli.profiling);
    if cli.profiling {
        profiling_reset();
    }

    if cli.debug {
        let workspace_root = match cli.tool_root.as_deref() {
            Some(raw) => PathBuf::from(raw),
            None => match std::env::current_dir() {
                Ok(dir) => dir,
                Err(e) => {
                    eprintln!("Tool workspace root: <failed to read current directory: {e}>");
                    PathBuf::from(".")
                }
            },
        };
        match workspace_root.canonicalize() {
            Ok(root) => eprintln!("Tool workspace root: {}", root.display()),
            Err(e) => eprintln!(
                "Tool workspace root: <cannot resolve '{}': {e}>",
                workspace_root.display()
            ),
        }
        let enabled_tools = cli.tool_enablement.enabled_tool_names();
        if enabled_tools.is_empty() {
            eprintln!("Allowed tools: none");
        } else {
            eprintln!("Allowed tools: {}", enabled_tools.join(", "));
        }
        eprintln!(
            "Parallel thresholds: matmul_min_rows={}, matmul_chunk_rows={}, attn_min_heads={}, qwen3next_min_heads={}",
            par_matmul_min_rows(),
            par_matmul_chunk_rows(),
            par_attn_min_heads(),
            par_qwen3next_min_heads()
        );
        #[cfg(target_arch = "aarch64")]
        eprintln!(
            "AArch64 prefetch: matmul_prefetch_rows={}",
            aarch64_matmul_prefetch_rows()
        );
        eprintln!("KV cache mode request: {:?}", kv_cache_mode());
    }

    let mut runtime = generation::ModelRuntime::load(&cli)?;
    match cli.mode {
        CliOperationMode::Oneshot => {
            run_oneshot_mode(&mut runtime, &cli)?;
        }
        CliOperationMode::Repl => {
            run_repl_mode(&mut runtime, &cli)?;
        }
    }

    if cli.profiling {
        print_profile_report();
    }
    if cli.show_timings {
        eprintln!(
            "overall runtime: {:.3}s",
            run_started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

fn run_oneshot_mode(
    runtime: &mut generation::ModelRuntime,
    cli: &CliOptions,
) -> Result<(), String> {
    if cli.tools_enabled {
        if !cli.images.is_empty() || !cli.videos.is_empty() || !cli.audios.is_empty() {
            return Err(
                "`--image/--video/--audio` are not supported together with tools mode yet"
                    .to_string(),
            );
        }
        return agent::run_agent_loop(runtime, cli, &cli.prompt);
    }

    let images = validate_media_paths(
        &cli.images,
        "image",
        MAX_IMAGES,
        MAX_IMAGE_BYTES,
        Some(IMAGE_EXTENSIONS),
    )?;
    let videos = validate_media_paths(
        &cli.videos,
        "video",
        MAX_VIDEOS,
        MAX_VIDEO_BYTES,
        Some(VIDEO_EXTENSIONS),
    )?;
    let audios = validate_media_paths(&cli.audios, "audio", MAX_AUDIOS, MAX_AUDIO_BYTES, None)?;
    let request = build_generation_request(&cli.prompt, &cli.system_prompt, images, videos, audios);
    let _ = runtime.generate_request(&request, true)?;
    Ok(())
}

fn run_repl_mode(runtime: &mut generation::ModelRuntime, cli: &CliOptions) -> Result<(), String> {
    if !cli.images.is_empty() || !cli.videos.is_empty() || !cli.audios.is_empty() {
        return Err("`--image/--video/--audio` are not supported in repl mode yet".to_string());
    }

    eprintln!("Entering repl mode. Type /help for commands.");
    if !cli.prompt.trim().is_empty() {
        match handle_repl_command(cli, cli.prompt.trim()) {
            ReplCommandAction::Exit => return Ok(()),
            ReplCommandAction::Handled => {}
            ReplCommandAction::ModelPrompt(prompt) => {
                if let Err(e) = run_repl_turn(runtime, cli, prompt) {
                    eprintln!("Turn failed: {e}");
                }
            }
        }
    }

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|e| format!("stdout flush failed: {e}"))?;
        let mut line = String::new();
        let read = stdin
            .read_line(&mut line)
            .map_err(|e| format!("stdin read failed: {e}"))?;
        if read == 0 {
            break;
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            continue;
        }
        let completed = expand_repl_tab_completion(line);
        let input = completed.trim();
        match handle_repl_command(cli, input) {
            ReplCommandAction::Exit => break,
            ReplCommandAction::Handled => continue,
            ReplCommandAction::ModelPrompt(prompt) => {
                if let Err(e) = run_repl_turn(runtime, cli, prompt) {
                    eprintln!("Turn failed: {e}");
                }
            }
        }
    }

    Ok(())
}

enum ReplCommandAction<'a> {
    Exit,
    Handled,
    ModelPrompt(&'a str),
}

fn handle_repl_command<'a>(cli: &CliOptions, input: &'a str) -> ReplCommandAction<'a> {
    let Some(raw_cmd) = input.strip_prefix('/') else {
        return ReplCommandAction::ModelPrompt(input);
    };
    let cmd = raw_cmd
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match cmd.as_str() {
        "" => {
            eprintln!("Empty command. Type /help for commands.");
            ReplCommandAction::Handled
        }
        "help" => {
            print_repl_help();
            ReplCommandAction::Handled
        }
        "model" => {
            println!("{}", cli.model);
            ReplCommandAction::Handled
        }
        "exit" | "quit" => ReplCommandAction::Exit,
        _ => {
            eprintln!("Unknown command '/{}'. Type /help for commands.", cmd);
            ReplCommandAction::Handled
        }
    }
}

fn expand_repl_tab_completion(input: &str) -> String {
    let Some(raw_cmd) = input.strip_prefix('/') else {
        return input.to_string();
    };
    let Some(tab_pos) = raw_cmd.find('\t') else {
        return input.to_string();
    };
    let before_tab = &raw_cmd[..tab_pos];
    if before_tab.is_empty() || before_tab.contains(char::is_whitespace) {
        return input.replace('\t', "");
    }
    let mut matches = REPL_COMMANDS
        .iter()
        .copied()
        .filter(|cmd| cmd.starts_with(before_tab));
    let first = matches.next();
    let second = matches.next();
    if let (Some(cmd), None) = (first, second) {
        return format!("/{cmd}");
    }
    input.replace('\t', "")
}

fn print_repl_help() {
    println!("REPL commands:");
    println!("  /help   Show this help");
    println!("  /model  Show active model path");
    println!("  /exit   Exit REPL");
    println!("  /quit   Exit REPL");
    println!("  /e<Tab> expands to /exit");
}

fn run_repl_turn(
    runtime: &mut generation::ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
) -> Result<(), String> {
    if cli.tools_enabled {
        return agent::run_agent_loop(runtime, cli, prompt);
    }

    let request = build_generation_request(
        prompt,
        &cli.system_prompt,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let _ = runtime.generate_request(&request, true)?;
    Ok(())
}

fn validate_media_paths(
    paths: &[String],
    kind: &str,
    max_count: usize,
    max_bytes: u64,
    allowed_extensions: Option<&[&str]>,
) -> Result<Vec<String>, String> {
    if paths.len() > max_count {
        return Err(format!(
            "too many {kind} inputs: got {}, max allowed {max_count}",
            paths.len()
        ));
    }
    let mut validated = Vec::with_capacity(paths.len());
    for path in paths {
        let meta = fs::metadata(path).map_err(|e| format!("cannot read {kind} '{path}': {e}"))?;
        if !meta.is_file() {
            return Err(format!("{kind} path is not a file: {path}"));
        }
        if meta.len() == 0 {
            return Err(format!("{kind} file is empty: {path}"));
        }
        if meta.len() > max_bytes {
            return Err(format!(
                "{kind} file exceeds max size ({} bytes): {path}",
                max_bytes
            ));
        }
        if let Some(extensions) = allowed_extensions {
            let ext = Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !extensions.iter().any(|allowed| *allowed == ext) {
                let allowed_list = extensions.join(", ");
                return Err(format!(
                    "unsupported {kind} extension for '{path}'; allowed: {allowed_list}"
                ));
            }
        }
        validated.push(path.clone());
    }
    Ok(validated)
}

fn build_generation_request(
    prompt: &str,
    system_prompt: &str,
    images: Vec<String>,
    videos: Vec<String>,
    audios: Vec<String>,
) -> GenerationRequest {
    let mut parts = Vec::with_capacity(1 + images.len() + videos.len() + audios.len());
    for path in images {
        parts.push(ContentPart::Image(MediaRef { path }));
    }
    for path in videos {
        parts.push(ContentPart::Video(MediaRef { path }));
    }
    for path in audios {
        parts.push(ContentPart::Audio(MediaRef { path }));
    }
    parts.push(ContentPart::Text(prompt.to_string()));
    GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
    }
}

#[cfg(test)]
mod tests {
    use super::expand_repl_tab_completion;

    #[test]
    fn repl_tab_completion_completes_exit() {
        assert_eq!(expand_repl_tab_completion("/e\t"), "/exit");
    }

    #[test]
    fn repl_tab_completion_completes_model() {
        assert_eq!(expand_repl_tab_completion("/mo\t"), "/model");
    }

    #[test]
    fn repl_tab_completion_ambiguous_prefix_keeps_slash_only() {
        assert_eq!(expand_repl_tab_completion("/\t"), "/");
    }
}
