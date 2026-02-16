use crate::cli::AgentToolEnablement;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_WRITE_BYTES: usize = 256 * 1024;
const MAX_LIST_ENTRIES: usize = 200;
const MAX_SHELL_ARGS: usize = 64;
const MAX_SHELL_ARG_BYTES: usize = 4096;
const MAX_SHELL_OUTPUT_BYTES: usize = 128 * 1024;

pub(crate) struct ToolExecutor {
    root: PathBuf,
    tool_enablement: AgentToolEnablement,
    allow_shell_commands: Vec<String>,
}

impl ToolExecutor {
    pub(crate) fn new(
        tool_root: Option<&str>,
        tool_enablement: AgentToolEnablement,
        allow_shell_commands: &[String],
    ) -> Result<Self, String> {
        let root = match tool_root {
            Some(raw) => PathBuf::from(raw),
            None => std::env::current_dir()
                .map_err(|e| format!("cannot read current directory: {e}"))?,
        };
        let root = root
            .canonicalize()
            .map_err(|e| format!("cannot canonicalize tool root '{}': {e}", root.display()))?;
        if !root.is_dir() {
            return Err(format!("tool root is not a directory: {}", root.display()));
        }
        let mut uniq = BTreeSet::new();
        for raw in allow_shell_commands {
            let normalized = normalize_shell_command(raw)
                .map_err(|e| format!("invalid shell command '{raw}': {e}"))?;
            uniq.insert(normalized);
        }
        Ok(Self {
            root,
            tool_enablement,
            allow_shell_commands: uniq.into_iter().collect(),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn write_file_enabled(&self) -> bool {
        self.tool_enablement.write_file
    }

    pub(crate) fn shell_exec_enabled(&self) -> bool {
        self.tool_enablement.shell_exec
    }

    pub(crate) fn shell_list_allowed_enabled(&self) -> bool {
        self.tool_enablement.shell_list_allowed
    }

    pub(crate) fn shell_request_allowed_enabled(&self) -> bool {
        self.tool_enablement.shell_request_allowed
    }

    pub(crate) fn has_any_filesystem_tool(&self) -> bool {
        self.tool_enablement.read_file || self.tool_enablement.list_dir || self.write_file_enabled()
    }

    pub(crate) fn enabled_tool_names(&self) -> Vec<&'static str> {
        let mut tools = Vec::new();
        if self.tool_enablement.read_file {
            tools.push("read_file");
        }
        if self.tool_enablement.write_file {
            tools.push("write_file");
        }
        if self.tool_enablement.list_dir {
            tools.push("list_dir");
        }
        if self.tool_enablement.shell_list_allowed {
            tools.push("shell_list_allowed");
        }
        if self.tool_enablement.shell_exec {
            tools.push("shell_exec");
        }
        if self.tool_enablement.shell_request_allowed {
            tools.push("shell_request_allowed");
        }
        tools
    }

    pub(crate) fn shell_allowed_commands(&self) -> &[String] {
        &self.allow_shell_commands
    }

    pub(crate) fn execute(&self, tool: &str, args: &Value) -> Result<Value, String> {
        match tool {
            "read_file" => {
                if !self.tool_enablement.read_file {
                    return Err(
                        "tool 'read_file' is disabled by config ([tools].read_file=false)"
                            .to_string(),
                    );
                }
                self.read_file(args)
            }
            "write_file" => {
                if !self.tool_enablement.write_file {
                    return Err(
                        "tool 'write_file' is disabled by config ([tools].write_file=false)"
                            .to_string(),
                    );
                }
                self.write_file(args)
            }
            "list_dir" => {
                if !self.tool_enablement.list_dir {
                    return Err(
                        "tool 'list_dir' is disabled by config ([tools].list_dir=false)"
                            .to_string(),
                    );
                }
                self.list_dir(args)
            }
            "shell_list_allowed" => {
                if !self.tool_enablement.shell_list_allowed {
                    return Err(
                        "tool 'shell_list_allowed' is disabled by config ([tools].shell_list_allowed=false)"
                            .to_string(),
                    );
                }
                Ok(self.shell_list_allowed())
            }
            "shell_exec" => {
                if !self.tool_enablement.shell_exec {
                    return Err(
                        "tool 'shell_exec' is disabled by config ([tools].shell_exec=false)"
                            .to_string(),
                    );
                }
                self.shell_exec(args)
            }
            "shell_request_allowed" => {
                if !self.tool_enablement.shell_request_allowed {
                    return Err("tool 'shell_request_allowed' is disabled by config ([tools].shell_request_allowed=false)".to_string());
                }
                self.shell_request_allowed(args)
            }
            _ => Err(format!("unknown tool '{tool}'")),
        }
    }

    fn resolve_existing_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        if raw_path.is_empty() {
            return Err("path cannot be empty".to_string());
        }
        let path = Path::new(raw_path);
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        let canonical = joined
            .canonicalize()
            .map_err(|e| format!("cannot resolve path '{}': {e}", joined.display()))?;
        if !canonical.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes tool root '{}'",
                canonical.display(),
                self.root.display()
            ));
        }
        Ok(canonical)
    }

    fn resolve_write_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        if raw_path.is_empty() {
            return Err("path cannot be empty".to_string());
        }
        let path = Path::new(raw_path);
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        let parent = joined
            .parent()
            .ok_or_else(|| format!("cannot determine parent for '{}'", joined.display()))?;
        let canonical_parent = parent.canonicalize().map_err(|e| {
            format!(
                "cannot resolve parent directory '{}': {e}",
                parent.display()
            )
        })?;
        if !canonical_parent.starts_with(&self.root) {
            return Err(format!(
                "path '{}' escapes tool root '{}'",
                joined.display(),
                self.root.display()
            ));
        }
        let file_name = joined
            .file_name()
            .ok_or_else(|| format!("cannot determine file name for '{}'", joined.display()))?;
        let final_path = canonical_parent.join(file_name);
        if final_path.exists() {
            let md = fs::symlink_metadata(&final_path)
                .map_err(|e| format!("cannot stat '{}': {e}", final_path.display()))?;
            if md.file_type().is_symlink() {
                return Err(format!(
                    "refusing to write through symlink '{}'",
                    final_path.display()
                ));
            }
        }
        Ok(final_path)
    }

    fn read_file(&self, args: &Value) -> Result<Value, String> {
        let args: ReadFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid read_file args: {e}"))?;
        let path = self.resolve_existing_path(&args.path)?;
        if !path.is_file() {
            return Err(format!("not a regular file: {}", path.display()));
        }
        let read_limit = args.max_bytes.unwrap_or(MAX_READ_BYTES).min(MAX_READ_BYTES);
        let mut file =
            fs::File::open(&path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
        let mut buf = Vec::new();
        std::io::Read::by_ref(&mut file)
            .take((read_limit + 1) as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        let truncated = buf.len() > read_limit;
        if truncated {
            buf.truncate(read_limit);
        }
        let content = String::from_utf8_lossy(&buf).to_string();
        Ok(json!({
            "ok": true,
            "tool": "read_file",
            "path": path.display().to_string(),
            "bytes": buf.len(),
            "truncated": truncated,
            "content": content
        }))
    }

    fn write_file(&self, args: &Value) -> Result<Value, String> {
        let args: WriteFileArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid write_file args: {e}"))?;
        let path = self.resolve_write_path(&args.path)?;
        let bytes = args.content.as_bytes();
        if bytes.len() > MAX_WRITE_BYTES {
            return Err(format!(
                "write_file content too large: {} bytes > {} bytes limit",
                bytes.len(),
                MAX_WRITE_BYTES
            ));
        }

        let mut opts = OpenOptions::new();
        opts.create(true).write(true);
        if args.append.unwrap_or(false) {
            opts.append(true);
        } else {
            opts.truncate(true);
        }

        let mut file = opts
            .open(&path)
            .map_err(|e| format!("cannot open '{}' for write: {e}", path.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("cannot write '{}': {e}", path.display()))?;
        file.flush()
            .map_err(|e| format!("cannot flush '{}': {e}", path.display()))?;
        Ok(json!({
            "ok": true,
            "tool": "write_file",
            "path": path.display().to_string(),
            "bytes_written": bytes.len(),
            "append": args.append.unwrap_or(false)
        }))
    }

    fn list_dir(&self, args: &Value) -> Result<Value, String> {
        let args: ListDirArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid list_dir args: {e}"))?;
        let path = args.path.unwrap_or_else(|| ".".to_string());
        let path = self.resolve_existing_path(&path)?;
        if !path.is_dir() {
            return Err(format!("not a directory: {}", path.display()));
        }
        let mut entries = Vec::new();
        for entry in
            fs::read_dir(&path).map_err(|e| format!("cannot list '{}': {e}", path.display()))?
        {
            let entry = entry.map_err(|e| format!("cannot read directory entry: {e}"))?;
            let ft = entry
                .file_type()
                .map_err(|e| format!("cannot read file type: {e}"))?;
            let kind = if ft.is_dir() {
                "dir"
            } else if ft.is_file() {
                "file"
            } else if ft.is_symlink() {
                "symlink"
            } else {
                "other"
            };
            entries.push(json!({
                "name": entry.file_name().to_string_lossy().to_string(),
                "kind": kind
            }));
            if entries.len()
                >= args
                    .max_entries
                    .unwrap_or(MAX_LIST_ENTRIES)
                    .min(MAX_LIST_ENTRIES)
            {
                break;
            }
        }
        entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        Ok(json!({
            "ok": true,
            "tool": "list_dir",
            "path": path.display().to_string(),
            "entries": entries
        }))
    }

    fn is_shell_command_allowed(&self, command: &str) -> bool {
        self.allow_shell_commands
            .binary_search_by(|allowed| allowed.as_str().cmp(command))
            .is_ok()
    }

    fn shell_list_allowed(&self) -> Value {
        json!({
            "ok": true,
            "tool": "shell_list_allowed",
            "shell_allowed_commands": self.allow_shell_commands.clone(),
            "internal_tool_status": {
                "read_file": self.tool_enablement.read_file,
                "list_dir": self.tool_enablement.list_dir,
                "write_file": self.write_file_enabled(),
                "shell_list_allowed": self.tool_enablement.shell_list_allowed,
                "shell_exec": self.tool_enablement.shell_exec,
                "shell_request_allowed": self.tool_enablement.shell_request_allowed
            }
        })
    }

    fn shell_exec(&self, args: &Value) -> Result<Value, String> {
        let args: RunShellArgs = serde_json::from_value(args.clone())
            .map_err(|e| {
                format!(
                    "invalid shell_exec args: {e}. expected object like {{\"command\":\"<allowed>\",\"args\":[...],\"cwd\":\".\",\"max_output_bytes\":131072}} (aliases accepted: cmd, argv, workdir)"
                )
            })?;
        let (command, argv) = normalize_shell_exec_invocation(&args.command, args.args)?;
        if !self.is_shell_command_allowed(&command) {
            let allowed = if self.allow_shell_commands.is_empty() {
                "<none>".to_string()
            } else {
                self.allow_shell_commands.join(", ")
            };
            return Err(format!(
                "command '{}' is not allowed. allowed commands: {}",
                command, allowed
            ));
        }

        if argv.len() > MAX_SHELL_ARGS {
            return Err(format!(
                "too many shell_exec args: {} > {}",
                argv.len(),
                MAX_SHELL_ARGS
            ));
        }
        for (idx, arg) in argv.iter().enumerate() {
            if arg.as_bytes().contains(&0) {
                return Err(format!("shell_exec arg {} contains NUL byte", idx));
            }
            if arg.len() > MAX_SHELL_ARG_BYTES {
                return Err(format!(
                    "shell_exec arg {} too large: {} bytes > {} bytes limit",
                    idx,
                    arg.len(),
                    MAX_SHELL_ARG_BYTES
                ));
            }
        }

        let cwd = if let Some(raw_cwd) = args.cwd {
            let dir = self.resolve_existing_path(&raw_cwd)?;
            if !dir.is_dir() {
                return Err(format!(
                    "shell_exec cwd is not a directory: {}",
                    dir.display()
                ));
            }
            dir
        } else {
            self.root.clone()
        };
        let output_limit = args
            .max_output_bytes
            .unwrap_or(MAX_SHELL_OUTPUT_BYTES)
            .min(MAX_SHELL_OUTPUT_BYTES);

        if command == "cwd" {
            if !argv.is_empty() {
                return Err("shell_exec command 'cwd' does not accept args".to_string());
            }
            let stdout_raw = format!("{}\n", cwd.display()).into_bytes();
            let (stdout, stdout_truncated) = truncate_output(&stdout_raw, output_limit);
            return Ok(json!({
                "ok": true,
                "tool": "shell_exec",
                "command": command,
                "args": argv,
                "cwd": cwd.display().to_string(),
                "exit_code": 0,
                "stdout_bytes": stdout_raw.len(),
                "stderr_bytes": 0,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": false,
                "stdout": stdout,
                "stderr": ""
            }));
        }

        let output = Command::new(&command)
            .args(&argv)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("shell_exec failed for '{}': {e}", command))?;

        let (stdout, stdout_truncated) = truncate_output(&output.stdout, output_limit);
        let (stderr, stderr_truncated) = truncate_output(&output.stderr, output_limit);
        Ok(json!({
            "ok": output.status.success(),
            "tool": "shell_exec",
            "command": command,
            "args": argv,
            "cwd": cwd.display().to_string(),
            "exit_code": output.status.code(),
            "stdout_bytes": output.stdout.len(),
            "stderr_bytes": output.stderr.len(),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "stdout": stdout,
            "stderr": stderr
        }))
    }

    fn shell_request_allowed(&self, args: &Value) -> Result<Value, String> {
        let args: RequestShellAllowedArgs = serde_json::from_value(args.clone())
            .map_err(|e| format!("invalid shell_request_allowed args: {e}"))?;
        let command = normalize_shell_command(&args.command)
            .map_err(|e| format!("invalid command request: {e}"))?;
        let already_allowed = self.is_shell_command_allowed(&command);
        let status = if already_allowed {
            "already_allowed"
        } else {
            "needs_user_approval"
        };
        let hint = if already_allowed {
            "Command is already allowed. Use shell_exec to execute it.".to_string()
        } else {
            format!(
                "Ask the operator to add --allow-shell-command {} (or GGUF_ALLOW_SHELL_COMMANDS).",
                command
            )
        };
        Ok(json!({
            "ok": true,
            "tool": "shell_request_allowed",
            "command": command,
            "reason": args.reason,
            "already_allowed": already_allowed,
            "status": status,
            "hint": hint
        }))
    }
}

fn normalize_shell_command(raw: &str) -> Result<String, String> {
    let command = raw.trim();
    if command.is_empty() {
        return Err("command cannot be empty".to_string());
    }
    if command.as_bytes().contains(&0) {
        return Err("command contains NUL byte".to_string());
    }
    if command
        .chars()
        .any(|c| c.is_ascii_whitespace() || c == '/' || c == '\\')
    {
        return Err(
            "command must be a bare executable name (no whitespace or path separators)".to_string(),
        );
    }
    Ok(command.to_string())
}

fn normalize_shell_exec_invocation(
    command_raw: &str,
    args: Option<Vec<String>>,
) -> Result<(String, Vec<String>), String> {
    let command_trimmed = command_raw.trim();
    let mut argv = args.unwrap_or_default();
    if argv.is_empty() && command_trimmed.chars().any(|c| c.is_ascii_whitespace()) {
        let mut parts = command_trimmed.split_whitespace();
        let head = parts
            .next()
            .ok_or_else(|| "invalid shell_exec command: command cannot be empty".to_string())?;
        let command = normalize_shell_command(head)
            .map_err(|e| format!("invalid shell_exec command: {e}"))?;
        argv.extend(parts.map(ToOwned::to_owned));
        return Ok((command, argv));
    }
    let command = normalize_shell_command(command_trimmed)
        .map_err(|e| format!("invalid shell_exec command: {e}"))?;
    Ok((command, argv))
}

fn truncate_output(bytes: &[u8], limit: usize) -> (String, bool) {
    let truncated = bytes.len() > limit;
    let slice = if truncated { &bytes[..limit] } else { bytes };
    (String::from_utf8_lossy(slice).to_string(), truncated)
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    max_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
    append: Option<bool>,
}

#[derive(Deserialize)]
struct ListDirArgs {
    path: Option<String>,
    max_entries: Option<usize>,
}

#[derive(Deserialize)]
struct RunShellArgs {
    #[serde(alias = "cmd", alias = "program", alias = "name")]
    command: String,
    #[serde(default, alias = "argv", alias = "arguments")]
    args: Option<Vec<String>>,
    #[serde(default, alias = "workdir", alias = "working_dir", alias = "dir")]
    cwd: Option<String>,
    #[serde(
        default,
        alias = "max_output",
        alias = "max_bytes",
        alias = "output_limit"
    )]
    max_output_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct RequestShellAllowedArgs {
    command: String,
    reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{normalize_shell_exec_invocation, RunShellArgs};

    #[test]
    fn run_shell_args_accepts_cmd_and_argv_aliases() {
        let raw = serde_json::json!({
            "cmd": "ls",
            "argv": ["-la"],
            "workdir": ".",
            "max_output": 1024
        });
        let parsed: RunShellArgs = serde_json::from_value(raw).expect("valid alias payload");
        assert_eq!(parsed.command, "ls");
        assert_eq!(parsed.args, Some(vec!["-la".to_string()]));
        assert_eq!(parsed.cwd, Some(".".to_string()));
        assert_eq!(parsed.max_output_bytes, Some(1024));
    }

    #[test]
    fn normalize_shell_exec_invocation_splits_command_line_when_args_missing() {
        let (command, args) =
            normalize_shell_exec_invocation("cargo check --release", None).expect("valid split");
        assert_eq!(command, "cargo");
        assert_eq!(args, vec!["check".to_string(), "--release".to_string()]);
    }
}
