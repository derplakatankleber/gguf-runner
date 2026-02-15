use serde::Deserialize;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_WRITE_BYTES: usize = 256 * 1024;
const MAX_LIST_ENTRIES: usize = 200;

pub(crate) struct ToolExecutor {
    root: PathBuf,
    allow_write: bool,
}

impl ToolExecutor {
    pub(crate) fn new(tool_root: Option<&str>, allow_write: bool) -> Result<Self, String> {
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
        Ok(Self { root, allow_write })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn allow_write(&self) -> bool {
        self.allow_write
    }

    pub(crate) fn execute(&self, tool: &str, args: &Value) -> Result<Value, String> {
        match tool {
            "read_file" => self.read_file(args),
            "write_file" => self.write_file(args),
            "list_dir" => self.list_dir(args),
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
        if !self.allow_write {
            return Err("write_file is disabled (enable with --allow-write-tools)".to_string());
        }
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
