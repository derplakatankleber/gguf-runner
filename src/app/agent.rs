use crate::app::generation::ModelRuntime;
use crate::app::tools::ToolExecutor;
use crate::cli::CliOptions;
use serde::Deserialize;
use serde_json::{json, Value};

struct AgentMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AgentResponse {
    Final { content: String },
    ToolCall { tool: String, args: Option<Value> },
}

pub(crate) fn run_agent_loop(runtime: &mut ModelRuntime, cli: &CliOptions) -> Result<(), String> {
    let tool_exec = ToolExecutor::new(cli.tool_root.as_deref(), cli.allow_write_tools)?;
    let system_prompt = build_agent_system_prompt(&cli.system_prompt, &tool_exec);
    let mut transcript = vec![AgentMessage {
        role: "user",
        content: cli.prompt.clone(),
    }];
    let mut tool_calls = 0usize;
    let mut protocol_failures = 0usize;
    let max_protocol_failures = 3usize;
    let max_turns = cli.max_tool_calls.saturating_mul(3).saturating_add(8);
    let require_tool_before_final = prompt_requires_filesystem(&cli.prompt);

    for _turn in 0..max_turns {
        let prompt = build_turn_prompt(&transcript);
        let raw = runtime.generate_text(&prompt, &system_prompt, false)?;
        match parse_agent_response(&raw) {
            Ok(AgentResponse::Final { content }) => {
                if require_tool_before_final && tool_calls == 0 {
                    protocol_failures += 1;
                    if protocol_failures > max_protocol_failures {
                        return Err(
                            "model returned final response without using any filesystem tool"
                                .to_string(),
                        );
                    }
                    transcript.push(AgentMessage {
                        role: "assistant",
                        content,
                    });
                    transcript.push(AgentMessage {
                        role: "user",
                        content: "You must call at least one filesystem tool before finalizing this request."
                            .to_string(),
                    });
                    continue;
                }
                println!("{content}");
                return Ok(());
            }
            Ok(AgentResponse::ToolCall { tool, args }) => {
                if tool_calls >= cli.max_tool_calls {
                    return Err(format!(
                        "max-tool-calls ({}) reached before final response",
                        cli.max_tool_calls
                    ));
                }
                tool_calls += 1;
                let args = args.unwrap_or_else(|| json!({}));
                let tool_result = match tool_exec.execute(&tool, &args) {
                    Ok(v) => v,
                    Err(e) => json!({
                        "ok": false,
                        "tool": tool,
                        "error": e
                    }),
                };
                let tool_result_text =
                    serde_json::to_string_pretty(&tool_result).map_err(|e| e.to_string())?;
                transcript.push(AgentMessage {
                    role: "assistant",
                    content: format!(
                        "tool_call\n{}",
                        json!({
                            "tool": tool,
                            "args": args
                        })
                    ),
                });
                transcript.push(AgentMessage {
                    role: "tool",
                    content: tool_result_text,
                });
            }
            Err(parse_error) => {
                protocol_failures += 1;
                if protocol_failures > max_protocol_failures {
                    let raw_preview = raw.trim().chars().take(240).collect::<String>();
                    return Err(format!(
                        "model did not follow agent JSON protocol after {} attempts: {}. Last output: {}",
                        max_protocol_failures, parse_error, raw_preview
                    ));
                }
                transcript.push(AgentMessage {
                    role: "assistant",
                    content: raw.trim().to_string(),
                });
                transcript.push(AgentMessage {
                    role: "user",
                    content: "Protocol error: reply with exactly one JSON object only. Use either {\"type\":\"tool_call\",\"tool\":\"...\",\"args\":{...}} or {\"type\":\"final\",\"content\":\"...\"}."
                        .to_string(),
                });
            }
        }
    }

    Err("agent loop reached maximum turns without final response".to_string())
}

fn build_agent_system_prompt(base_system_prompt: &str, tool_exec: &ToolExecutor) -> String {
    let write_state = if tool_exec.allow_write() {
        "enabled"
    } else {
        "disabled"
    };
    format!(
        "{base_system_prompt}\n\n\
You are running with host tools. \
Always respond with exactly one JSON object and no surrounding markdown.\n\
Allowed response schemas:\n\
1) Tool call:\n\
{{\"type\":\"tool_call\",\"tool\":\"read_file|write_file|list_dir\",\"args\":{{...}}}}\n\
2) Final answer:\n\
{{\"type\":\"final\",\"content\":\"...\"}}\n\
Rules:\n\
- Use tools when you need filesystem state.\n\
- If the user asks about files, code, directories, or repository contents, call a filesystem tool before answering.\n\
- Keep tool arguments minimal and valid JSON.\n\
- If a tool fails, adjust and retry.\n\
- Return `type=final` when done.\n\
Tool constraints:\n\
- tool_root: {}\n\
- write_file: {}\n\
- max read/write payload per call: 262144 bytes\n",
        tool_exec.root().display(),
        write_state
    )
}

fn build_turn_prompt(transcript: &[AgentMessage]) -> String {
    let mut out = String::from("Transcript:\n");
    for msg in transcript {
        out.push_str("<<<");
        out.push_str(msg.role);
        out.push_str(">>>\n");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out.push_str("Respond with one JSON object now.");
    out
}

fn parse_agent_response(raw: &str) -> Result<AgentResponse, String> {
    if let Ok(parsed) = serde_json::from_str::<AgentResponse>(raw.trim()) {
        return Ok(parsed);
    }

    let value = extract_first_json_value(raw)
        .ok_or_else(|| "no JSON object found in model output".to_string())?;
    if let Ok(parsed) = serde_json::from_value::<AgentResponse>(value.clone()) {
        return Ok(parsed);
    }
    parse_agent_response_from_value(value)
}

fn parse_agent_response_from_value(value: Value) -> Result<AgentResponse, String> {
    if let Some(content) = value.get("content").and_then(Value::as_str) {
        if value.get("type").and_then(Value::as_str) == Some("final") || value.get("type").is_none()
        {
            return Ok(AgentResponse::Final {
                content: content.to_string(),
            });
        }
    }
    if let Some(tool) = value.get("tool").and_then(Value::as_str) {
        let args = value.get("args").cloned();
        if value.get("type").and_then(Value::as_str) == Some("tool_call")
            || value.get("type").is_none()
        {
            return Ok(AgentResponse::ToolCall {
                tool: tool.to_string(),
                args,
            });
        }
    }
    Err("unsupported agent response payload".to_string())
}

fn extract_first_json_value(raw: &str) -> Option<Value> {
    for (idx, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        if let Some(v) = parse_first_json_value(&raw[idx..]) {
            if v.is_object() {
                return Some(v);
            }
        }
    }
    None
}

fn parse_first_json_value(s: &str) -> Option<Value> {
    let mut de = serde_json::Deserializer::from_str(s);
    Value::deserialize(&mut de).ok()
}

fn prompt_requires_filesystem(prompt: &str) -> bool {
    let p = prompt.to_ascii_lowercase();
    let hints = [
        "file",
        "directory",
        "folder",
        "src/",
        ".rs",
        ".toml",
        "inspect",
        "read",
        "list",
        "repo",
        "repository",
        "codebase",
    ];
    hints.iter().any(|h| p.contains(h))
}
