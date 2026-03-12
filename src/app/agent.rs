use crate::app::events::{emit_runtime_event, RuntimeEvent, RuntimeEventCallback};
use crate::app::generation::ModelRuntime;
use crate::cli::{CliOptions, ShellCommandDescriptionSpec, ToolPromptSpec};
use crate::tools::ToolExecutor;
use crate::vendors::{ChatMessage, ChatRole};
use serde::Deserialize;
use serde_json::{json, Value};

struct AgentMessage {
    role: &'static str,
    content: String,
}

pub(crate) enum AgentRunEvent {
    Info(String),
    Output(String),
    Error(String),
}

pub(crate) struct AgentRunResult {
    pub(crate) events: Vec<AgentRunEvent>,
}

fn push_agent_event(
    events: &mut Vec<AgentRunEvent>,
    event: AgentRunEvent,
    callback: Option<&RuntimeEventCallback>,
) {
    let runtime_event = match &event {
        AgentRunEvent::Info(text) => RuntimeEvent::Info(text.clone()),
        AgentRunEvent::Output(text) => RuntimeEvent::Output(text.clone()),
        AgentRunEvent::Error(text) => RuntimeEvent::Error(text.clone()),
    };
    emit_runtime_event(callback, runtime_event);
    events.push(event);
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AgentResponse {
    Final { content: String },
    ToolCall { tool: String, args: Option<Value> },
}

pub(crate) fn run_agent_loop(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
) -> Result<(), String> {
    let result = run_agent_loop_collect(runtime, cli, prompt)?;
    for event in result.events {
        match event {
            AgentRunEvent::Info(text) => eprintln!("{text}"),
            AgentRunEvent::Output(text) => println!("{text}"),
            AgentRunEvent::Error(text) => eprintln!("{text}"),
        }
    }
    Ok(())
}

pub(crate) fn run_agent_loop_collect(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
) -> Result<AgentRunResult, String> {
    run_agent_loop_collect_with_history_callback(runtime, cli, &[], prompt, None)
}

pub(crate) fn run_agent_loop_collect_with_history_callback(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prior_chat_history: &[ChatMessage],
    prompt: &str,
    callback: Option<&RuntimeEventCallback>,
) -> Result<AgentRunResult, String> {
    let tool_exec = ToolExecutor::new(
        cli.tool_root.as_deref(),
        cli.tool_enablement.clone(),
        &cli.allow_shell_commands,
    )?;
    let system_prompt = build_agent_system_prompt(
        &cli.system_prompt,
        &tool_exec,
        &cli.tool_prompt_specs,
        &cli.shell_command_description_specs,
    );
    let mut transcript = Vec::new();
    for message in prior_chat_history {
        transcript.push(AgentMessage {
            role: match message.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            },
            content: message.content.clone(),
        });
    }
    transcript.push(AgentMessage {
        role: "user",
        content: prompt.to_string(),
    });
    let mut tool_calls = 0usize;
    let mut protocol_failures = 0usize;
    let mut events = Vec::new();
    let max_protocol_failures = 3usize;
    let max_turns = cli
        .max_tool_calls
        .saturating_mul(3)
        .saturating_add(8)
        .min(64);
    let require_tool_before_final =
        prompt_requires_filesystem(prompt) && tool_exec.has_any_filesystem_tool();

    for turn in 0..max_turns {
        if cli.debug {
            push_agent_event(
                &mut events,
                AgentRunEvent::Info(format!("Agent turn {}/{}", turn + 1, max_turns)),
                callback,
            );
        }
        let turn_prompt = build_turn_prompt(&transcript);
        let original_callback = runtime.runtime_event_callback();
        let filtered_callback = callback.map(|outer| {
            let outer = outer.clone();
            std::sync::Arc::new(move |event: RuntimeEvent| {
                if !matches!(event, RuntimeEvent::Output(_)) {
                    outer(event);
                }
            }) as RuntimeEventCallback
        });
        runtime.set_runtime_event_callback(filtered_callback);
        let raw = runtime.generate_text_for_agent(&turn_prompt, &system_prompt, false);
        runtime.set_runtime_event_callback(original_callback);
        let raw = raw?;
        if cli.debug {
            push_agent_event(
                &mut events,
                AgentRunEvent::Info(format!("Agent raw output bytes: {}", raw.len())),
                callback,
            );
        }
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
                        role: "user",
                        content: "You must call at least one filesystem tool before finalizing this request. Reply with exactly one JSON object: either a tool_call or final."
                            .to_string(),
                    });
                    continue;
                }
                push_agent_event(&mut events, AgentRunEvent::Output(content), callback);
                return Ok(AgentRunResult { events });
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
                let args_json =
                    serde_json::to_string(&args).unwrap_or_else(|_| "<invalid-json>".to_string());
                push_agent_event(
                    &mut events,
                    AgentRunEvent::Info(format!(
                        "Tool call [{}]: {} args={}",
                        tool_calls, tool, args_json
                    )),
                    callback,
                );
                let tool_result = match tool_exec.execute(&tool, &args) {
                    Ok(v) => v,
                    Err(e) => {
                        if tool == "shell_exec" {
                            push_agent_event(
                                &mut events,
                                AgentRunEvent::Error(format!("shell_exec error: {e}")),
                                callback,
                            );
                        }
                        json!({
                            "ok": false,
                            "tool": tool,
                            "error": e
                        })
                    }
                };
                if tool == "shell_exec"
                    && tool_result.get("ok").and_then(Value::as_bool) == Some(false)
                    && tool_result.get("exit_code").is_some()
                {
                    let exit_code = tool_result
                        .get("exit_code")
                        .and_then(Value::as_i64)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let stderr = tool_result
                        .get("stderr")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let stderr_preview = stderr.trim().chars().take(240).collect::<String>();
                    if stderr_preview.is_empty() {
                        push_agent_event(
                            &mut events,
                            AgentRunEvent::Error(format!(
                                "shell_exec failed (exit_code={exit_code})"
                            )),
                            callback,
                        );
                    } else {
                        push_agent_event(
                            &mut events,
                            AgentRunEvent::Error(format!(
                                "shell_exec failed (exit_code={exit_code}): {}",
                                stderr_preview
                            )),
                            callback,
                        );
                    }
                }
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
                if cli.debug {
                    let preview = raw.trim().chars().take(180).collect::<String>();
                    push_agent_event(
                        &mut events,
                        AgentRunEvent::Error(format!(
                            "Agent protocol parse error: {} | preview: {}",
                            parse_error, preview
                        )),
                        callback,
                    );
                }
                let fallback = raw.trim();
                let fallback_looks_like_json = fallback.starts_with('{');
                if tool_calls > 0
                    && !fallback.is_empty()
                    && !fallback_looks_like_json
                    && looks_like_reasonable_fallback_text(fallback)
                {
                    if cli.debug {
                        push_agent_event(
                            &mut events,
                            AgentRunEvent::Info(format!(
                                "Agent protocol fallback after tool call(s): {}",
                                parse_error
                            )),
                            callback,
                        );
                    }
                    push_agent_event(
                        &mut events,
                        AgentRunEvent::Output(fallback.to_string()),
                        callback,
                    );
                    return Ok(AgentRunResult { events });
                }
                protocol_failures += 1;
                if protocol_failures > max_protocol_failures {
                    if !require_tool_before_final {
                        let fallback = raw.trim();
                        if !fallback.is_empty() {
                            push_agent_event(
                                &mut events,
                                AgentRunEvent::Output(fallback.to_string()),
                                callback,
                            );
                            return Ok(AgentRunResult { events });
                        }
                    }
                    let raw_preview = raw.trim().chars().take(240).collect::<String>();
                    return Err(format!(
                        "model did not follow agent JSON protocol after {} attempts: {}. Last output: {}",
                        max_protocol_failures, parse_error, raw_preview
                    ));
                }
                let mut protocol_msg = "Protocol error: reply with exactly one JSON object only. Use either {\"type\":\"tool_call\",\"tool\":\"...\",\"args\":{...}} or {\"type\":\"final\",\"content\":\"...\"}."
                    .to_string();
                if require_tool_before_final && tool_calls == 0 {
                    protocol_msg.push_str(
                        " For this request you must call a filesystem tool before final.",
                    );
                }
                transcript.push(AgentMessage {
                    role: "user",
                    content: protocol_msg,
                });
            }
        }
    }

    Err("agent loop reached maximum turns without final response".to_string())
}

fn build_agent_system_prompt(
    base_system_prompt: &str,
    tool_exec: &ToolExecutor,
    tool_prompt_specs: &[ToolPromptSpec],
    shell_command_description_specs: &[ShellCommandDescriptionSpec],
) -> String {
    let _metadata_bytes: usize = tool_prompt_specs
        .iter()
        .map(|s| s.name.len() + s.description.len() + s.when_to_use.len())
        .sum::<usize>()
        + shell_command_description_specs
            .iter()
            .map(|s| s.command.len() + s.description.len())
            .sum::<usize>();
    let write_state = if tool_exec.write_file_enabled() {
        "enabled"
    } else {
        "disabled"
    };
    let enabled_tools = tool_exec.enabled_tool_names();
    let allowed_tool_names = if enabled_tools.is_empty() {
        "<none>".to_string()
    } else {
        enabled_tools.join("|")
    };
    let shell_allowed_commands = if tool_exec.shell_allowed_commands().is_empty() {
        "<none>".to_string()
    } else {
        tool_exec.shell_allowed_commands().join(", ")
    };
    let tool_rules = render_tool_rules(tool_exec);
    format!(
        "{base_system_prompt}\n\n\
You are running with host tools.\n\
Always respond with exactly one JSON object and no surrounding markdown.\n\
Use compact JSON on a single line, with keys in the exact order shown below.\n\
Allowed response schemas:\n\
1) Tool call:\n\
{{\"type\":\"tool_call\",\"tool\":\"{}\",\"args\":{{...}}}}\n\
2) Final answer:\n\
{{\"type\":\"final\",\"content\":\"...\"}}\n\
Rules:\n\
{}\n\
Runtime constraints:\n\
- tool_root: {}\n\
- write_file: {}\n\
- shell allowed commands: {}\n\
- max read/write payload per call: 262144 bytes\n\
Output rules:\n\
- If the user asks about files/repo contents and filesystem tools are enabled, call a filesystem tool before final.\n\
- If a tool call fails due args shape, fix the args and retry.\n\
- Avoid prose outside the single JSON object.\n",
        allowed_tool_names,
        tool_rules,
        tool_exec.root().display(),
        write_state,
        shell_allowed_commands
    )
}

fn looks_like_reasonable_fallback_text(text: &str) -> bool {
    if text.len() < 24 || text.contains('\0') {
        return false;
    }
    let mut total = 0usize;
    let mut humanish = 0usize;
    let mut alpha = 0usize;
    for c in text.chars() {
        total += 1;
        if c.is_ascii_alphabetic() {
            alpha += 1;
            humanish += 1;
            continue;
        }
        if c.is_ascii_whitespace() || c.is_ascii_punctuation() || c.is_ascii_digit() {
            humanish += 1;
        }
    }
    if total == 0 {
        return false;
    }
    let humanish_ratio = humanish as f32 / total as f32;
    let alpha_ratio = alpha as f32 / total as f32;
    humanish_ratio > 0.92 && alpha_ratio > 0.08
}

fn render_tool_rules(tool_exec: &ToolExecutor) -> String {
    let mut rules = vec![
        "- Use tools when you need filesystem state.".to_string(),
        "- If the user asks about files, code, directories, or repository contents, call a filesystem tool before answering when such tools are enabled.".to_string(),
        "- Keep tool arguments minimal and valid JSON.".to_string(),
        "- If a tool fails, adjust and retry.".to_string(),
        "- Return `type=final` when done.".to_string(),
    ];
    if tool_exec.shell_exec_enabled() {
        rules.push(
            "- `shell_exec` can only execute commands already in the shell allowed list."
                .to_string(),
        );
        rules.push(
            "- `shell_exec` args must include a command key (or alias cmd). Example: {\"type\":\"tool_call\",\"tool\":\"shell_exec\",\"args\":{\"command\":\"ls\",\"args\":[\"-la\"]}}."
                .to_string(),
        );
    }
    if tool_exec.shell_list_allowed_enabled() {
        rules.push(
            "- Use `shell_list_allowed` when you need to inspect shell allowed commands or internal tool status.".to_string(),
        );
    }
    if tool_exec.shell_request_allowed_enabled() {
        rules.push("- If a needed command is missing from the shell allowed list, call `shell_request_allowed` with command + reason.".to_string());
    }
    rules.join("\n")
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

    let mut saw_json_object = false;
    for value in extract_json_objects(raw) {
        saw_json_object = true;
        if let Ok(parsed) = serde_json::from_value::<AgentResponse>(value.clone()) {
            return Ok(parsed);
        }
        if let Ok(parsed) = parse_agent_response_from_value(value) {
            return Ok(parsed);
        }
    }
    if saw_json_object {
        Err("unsupported agent response payload".to_string())
    } else {
        Err("no JSON object found in model output".to_string())
    }
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

fn extract_json_objects(raw: &str) -> Vec<Value> {
    let mut values = Vec::new();
    for (idx, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        if let Some(v) = parse_first_json_value(&raw[idx..]) {
            if v.is_object() {
                values.push(v);
            }
        }
    }
    values
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

#[cfg(test)]
mod tests {
    use super::looks_like_reasonable_fallback_text;

    #[test]
    fn fallback_text_guard_rejects_numeric_gibberish() {
        let gibberish = "000003 010 1000 200000101 01 600006 00000 000106 40 00016560";
        assert!(!looks_like_reasonable_fallback_text(gibberish));
    }
}
