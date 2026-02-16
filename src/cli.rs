use clap::Parser;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

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

fn parse_positive_f32(raw: &str) -> Result<f32, String> {
    let v = raw
        .parse::<f32>()
        .map_err(|e| format!("invalid value '{raw}': {e}"))?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}': expected > 0"))
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

fn parse_kv_cache_mode(raw: &str) -> Result<CliKvCacheMode, String> {
    let v = raw.trim();
    if v.eq_ignore_ascii_case("auto") {
        Ok(CliKvCacheMode::Auto)
    } else if v.eq_ignore_ascii_case("q8") {
        Ok(CliKvCacheMode::Q8)
    } else if v.eq_ignore_ascii_case("q4") {
        Ok(CliKvCacheMode::Q4)
    } else {
        Err(format!("invalid value '{raw}': expected one of auto/q8/q4"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CliKvCacheMode {
    Auto,
    Q8,
    Q4,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolPromptSpec {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) when_to_use: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ShellCommandDescriptionSpec {
    pub(crate) command: String,
    pub(crate) description: String,
}

#[derive(Clone, Debug)]
pub(crate) struct AgentToolEnablement {
    pub(crate) read_file: bool,
    pub(crate) list_dir: bool,
    pub(crate) write_file: bool,
    pub(crate) shell_list_allowed: bool,
    pub(crate) shell_exec: bool,
    pub(crate) shell_request_allowed: bool,
}

impl Default for AgentToolEnablement {
    fn default() -> Self {
        Self {
            read_file: true,
            list_dir: true,
            write_file: true,
            shell_list_allowed: true,
            shell_exec: true,
            shell_request_allowed: true,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct RunnerConfig {
    shell: Option<RunnerShellConfig>,
    tools: Option<RunnerToolsConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RunnerShellConfig {
    #[serde(alias = "md")]
    cmd: Option<BTreeMap<String, String>>,
    allowed_commands: Option<Vec<RunnerAllowedCommandEntry>>,
    allowed_command_descriptions: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Default, Deserialize)]
struct RunnerToolsConfig {
    read_file: Option<bool>,
    list_dir: Option<bool>,
    write_file: Option<bool>,
    shell_list_allowed: Option<bool>,
    shell_exec: Option<bool>,
    shell_request_allowed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RunnerAllowedCommandEntry {
    Name(String),
    Spec(RunnerAllowedCommandSpec),
}

#[derive(Debug, Deserialize)]
struct RunnerAllowedCommandSpec {
    name: String,
    description: Option<String>,
}

#[derive(Default)]
struct LoadedShellConfig {
    allowed_commands: Vec<String>,
    description_specs: Vec<ShellCommandDescriptionSpec>,
}

fn default_tool_prompt_specs() -> Vec<ToolPromptSpec> {
    vec![
        ToolPromptSpec {
            name: "read_file".to_string(),
            description: "Read UTF-8 file content under tool_root with a bounded byte limit."
                .to_string(),
            when_to_use: "Use when you need the contents of a specific file before reasoning or editing."
                .to_string(),
        },
        ToolPromptSpec {
            name: "list_dir".to_string(),
            description: "List directory entries under tool_root.".to_string(),
            when_to_use: "Use when you need to discover paths before reading or writing files."
                .to_string(),
        },
        ToolPromptSpec {
            name: "write_file".to_string(),
            description: "Write or append UTF-8 file content under tool_root.".to_string(),
            when_to_use: "Use only when the user explicitly requests file creation/modification."
                .to_string(),
        },
        ToolPromptSpec {
            name: "shell_list_allowed".to_string(),
            description: "Return currently enabled tools and allowed shell commands.".to_string(),
            when_to_use:
                "Use first when you are unsure which tool operations/commands are currently allowed."
                    .to_string(),
        },
        ToolPromptSpec {
            name: "shell_exec".to_string(),
            description:
                "Run an allowed external command with explicit argv (no shell expression). Args schema: {\"command\":\"<allowed>\",\"args\":[...],\"cwd\":\"optional\",\"max_output_bytes\":131072}. Supports built-in helper command `cwd`."
                    .to_string(),
            when_to_use:
                "Use when command output is needed and the command exists in allowed shell commands."
                    .to_string(),
        },
        ToolPromptSpec {
            name: "shell_request_allowed".to_string(),
            description:
                "Request operator approval for a command that is not currently in allowed shell commands."
                    .to_string(),
            when_to_use:
                "Use when shell_exec cannot run because a needed command is not currently allowed."
                    .to_string(),
        },
    ]
}

fn config_paths() -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".gguf-runner").join("config.toml");
        paths.push(p);
    }
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?;
    paths.push(cwd.join(".gguf-runner").join("config.toml"));
    Ok(paths)
}

fn load_shell_config_from_config() -> Result<LoadedShellConfig, String> {
    let mut allowed_commands = Vec::new();
    let mut descriptions = BTreeMap::new();
    for path in config_paths()? {
        let content = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot read config '{}': {e}", path.display())),
        };
        let parsed: RunnerConfig = toml::from_str(&content)
            .map_err(|e| format!("invalid config TOML '{}': {e}", path.display()))?;
        if let Some(shell) = parsed.shell {
            if let Some(cmd_entries) = shell.cmd {
                let (new_allowed_commands, new_descriptions) = parse_shell_cmd_entries(cmd_entries);
                allowed_commands = new_allowed_commands;
                descriptions = new_descriptions;
                continue;
            }
            if let Some(commands) = shell.allowed_commands {
                let (new_allowed_commands, legacy_descriptions) =
                    parse_allowed_command_entries(commands);
                allowed_commands = new_allowed_commands;
                descriptions = legacy_descriptions;
            }
            if let Some(extra_descriptions) = shell.allowed_command_descriptions {
                for (raw_command, raw_description) in extra_descriptions {
                    let Some(description) = normalize_description_text(raw_description) else {
                        continue;
                    };
                    for command in split_shell_command_names(&raw_command) {
                        descriptions.insert(command, description.clone());
                    }
                }
            }
        }
    }
    let description_specs = allowed_commands
        .iter()
        .filter_map(|command| {
            descriptions
                .get(command)
                .map(|description| ShellCommandDescriptionSpec {
                    command: command.clone(),
                    description: description.clone(),
                })
        })
        .collect();
    Ok(LoadedShellConfig {
        allowed_commands,
        description_specs,
    })
}

fn load_tool_enablement_from_config() -> Result<AgentToolEnablement, String> {
    let mut tool_enablement = AgentToolEnablement::default();
    for path in config_paths()? {
        let content = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("cannot read config '{}': {e}", path.display())),
        };
        let parsed: RunnerConfig = toml::from_str(&content)
            .map_err(|e| format!("invalid config TOML '{}': {e}", path.display()))?;
        if let Some(tools) = parsed.tools {
            if let Some(v) = tools.read_file {
                tool_enablement.read_file = v;
            }
            if let Some(v) = tools.list_dir {
                tool_enablement.list_dir = v;
            }
            if let Some(v) = tools.write_file {
                tool_enablement.write_file = v;
            }
            if let Some(v) = tools.shell_list_allowed {
                tool_enablement.shell_list_allowed = v;
            }
            if let Some(v) = tools.shell_exec {
                tool_enablement.shell_exec = v;
            }
            if let Some(v) = tools.shell_request_allowed {
                tool_enablement.shell_request_allowed = v;
            }
        }
    }
    Ok(tool_enablement)
}

fn parse_shell_cmd_entries(
    cmd_entries: BTreeMap<String, String>,
) -> (Vec<String>, BTreeMap<String, String>) {
    let mut raw_names = Vec::new();
    let mut descriptions = BTreeMap::new();
    for (raw_command, raw_description) in cmd_entries {
        let names = split_shell_command_names(&raw_command);
        raw_names.extend(names.iter().cloned());
        if let Some(description) = normalize_description_text(raw_description) {
            for name in names {
                descriptions.insert(name, description.clone());
            }
        }
    }
    let allowed_commands = normalize_shell_command_values(raw_names);
    (allowed_commands, descriptions)
}

fn normalize_shell_command_values<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut uniq = BTreeSet::new();
    for raw in values {
        for part in raw.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                uniq.insert(trimmed.to_string());
            }
        }
    }
    uniq.into_iter().collect()
}

fn parse_allowed_command_entries(
    entries: Vec<RunnerAllowedCommandEntry>,
) -> (Vec<String>, BTreeMap<String, String>) {
    let mut raw_names = Vec::new();
    let mut descriptions = BTreeMap::new();
    for entry in entries {
        match entry {
            RunnerAllowedCommandEntry::Name(name) => raw_names.push(name),
            RunnerAllowedCommandEntry::Spec(spec) => {
                let names = split_shell_command_names(&spec.name);
                raw_names.extend(names.iter().cloned());
                if let Some(description) = spec.description.and_then(normalize_description_text) {
                    for name in names {
                        descriptions.insert(name, description.clone());
                    }
                }
            }
        }
    }
    let allowed_commands = normalize_shell_command_values(raw_names);
    (allowed_commands, descriptions)
}

fn split_shell_command_names(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_description_text(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
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

    #[arg(
        long = "repeat-penalty",
        value_parser = parse_positive_f32,
        default_value_t = 1.0
    )]
    repeat_penalty: f32,

    #[arg(long = "repeat-last-n", default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long = "max-tokens", default_value_t = 0)]
    max_tokens: usize,

    #[arg(long = "context-size", default_value_t = 0)]
    context_size: usize,

    #[arg(
        long,
        env = "GGUF_RAYON_THREADS",
        value_parser = parse_positive_usize
    )]
    threads: Option<usize>,

    #[arg(long = "system-prompt", default_value = "You are a helpful assistant.")]
    system_prompt: String,

    #[arg(long)]
    agent: bool,

    #[arg(long = "tool-root", value_name = "path")]
    tool_root: Option<String>,

    #[arg(
        long = "allow-shell-command",
        value_name = "command",
        env = "GGUF_ALLOW_SHELL_COMMANDS",
        value_delimiter = ','
    )]
    allow_shell_commands: Vec<String>,

    #[arg(
        long = "max-tool-calls",
        value_parser = parse_positive_usize,
        default_value_t = 256
    )]
    max_tool_calls: usize,

    #[arg(long)]
    profiling: bool,

    #[arg(long = "show-tokens")]
    show_tokens: bool,

    #[arg(long = "show-timings")]
    show_timings: bool,

    #[arg(long)]
    debug: bool,

    #[arg(
        long = "kv-cache-mode",
        hide = true,
        env = "GGUF_KV_CACHE_MODE",
        value_parser = parse_kv_cache_mode
    )]
    kv_cache_mode: Option<CliKvCacheMode>,

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

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avxvnni",
        hide = true,
        env = "GGUF_X86_AVXVNNI",
        value_parser = parse_boolish
    )]
    x86_avxvnni: Option<bool>,

    #[cfg(target_arch = "x86_64")]
    #[arg(
        long = "x86-avx512vnni-q8",
        hide = true,
        env = "GGUF_X86_AVX512VNNI_Q8",
        value_parser = parse_boolish
    )]
    x86_avx512vnni_q8: Option<bool>,

    #[arg(
        long = "layer-debug",
        hide = true,
        env = "GGUF_LAYER_DEBUG",
        value_parser = parse_boolish
    )]
    layer_debug: Option<bool>,

    #[arg(long = "layer-debug-pos", hide = true, env = "GGUF_LAYER_DEBUG_POS")]
    layer_debug_pos: Option<usize>,
}

pub(crate) struct CliOptions {
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) url: Option<String>,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) context_size: usize,
    pub(crate) threads: Option<usize>,
    pub(crate) system_prompt: String,
    pub(crate) agent: bool,
    pub(crate) tool_root: Option<String>,
    pub(crate) tool_enablement: AgentToolEnablement,
    pub(crate) allow_shell_commands: Vec<String>,
    pub(crate) shell_command_description_specs: Vec<ShellCommandDescriptionSpec>,
    pub(crate) tool_prompt_specs: Vec<ToolPromptSpec>,
    pub(crate) max_tool_calls: usize,
    pub(crate) profiling: bool,
    pub(crate) show_tokens: bool,
    pub(crate) show_timings: bool,
    pub(crate) debug: bool,
    pub(crate) kv_cache_mode: Option<CliKvCacheMode>,
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
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avxvnni: Option<bool>,
    #[cfg(target_arch = "x86_64")]
    pub(crate) x86_avx512vnni_q8: Option<bool>,
    pub(crate) layer_debug: Option<bool>,
    pub(crate) layer_debug_pos: Option<usize>,
}

impl CliOptions {
    pub(crate) fn parse() -> Result<Self, String> {
        let cli = Cli::try_parse().map_err(|e| e.to_string())?;
        let tool_prompt_specs = default_tool_prompt_specs();
        let loaded_shell = if cli.agent {
            load_shell_config_from_config()?
        } else {
            LoadedShellConfig::default()
        };
        let tool_enablement = if cli.agent {
            load_tool_enablement_from_config()?
        } else {
            AgentToolEnablement::default()
        };
        let LoadedShellConfig {
            allowed_commands: mut allow_shell_commands,
            description_specs: shell_command_description_specs,
        } = loaded_shell;
        allow_shell_commands.extend(cli.allow_shell_commands);
        let allow_shell_commands = normalize_shell_command_values(allow_shell_commands);

        Ok(Self {
            model: cli.model,
            prompt: cli.prompt,
            url: cli.url,
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            repeat_penalty: cli.repeat_penalty,
            repeat_last_n: cli.repeat_last_n,
            max_tokens: cli.max_tokens,
            context_size: cli.context_size,
            threads: cli.threads,
            system_prompt: cli.system_prompt,
            agent: cli.agent,
            tool_root: cli.tool_root,
            tool_enablement,
            allow_shell_commands,
            shell_command_description_specs,
            tool_prompt_specs,
            max_tool_calls: cli.max_tool_calls,
            profiling: cli.profiling,
            show_tokens: cli.show_tokens,
            show_timings: cli.show_timings,
            debug: cli.debug,
            kv_cache_mode: cli.kv_cache_mode,
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
            #[cfg(target_arch = "x86_64")]
            x86_avxvnni: cli.x86_avxvnni,
            #[cfg(target_arch = "x86_64")]
            x86_avx512vnni_q8: cli.x86_avx512vnni_q8,
            layer_debug: cli.layer_debug,
            layer_debug_pos: cli.layer_debug_pos,
        })
    }
}
