use crate::app::agent;
use crate::app::events::{
    emit_runtime_event, RuntimeEvent, RuntimeEventCallback, RuntimePhase, RuntimeProgress,
};
use crate::app::generation::ModelRuntime;
use crate::app::{collect_debug_banner_lines, expand_repl_tab_completion, handle_repl_command};
use crate::cli::CliOptions;
use crate::vendors::{ChatMessage, ChatRole};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Alignment;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::cmp::min;
use std::io::{self, Stdout};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplLineKind {
    System,
    User,
    Assistant,
    Debug,
    Error,
}

struct ReplLine {
    kind: ReplLineKind,
    text: String,
}

struct ReplApp {
    input: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    messages: Vec<ReplLine>,
    chat_history: Vec<ChatMessage>,
    pending_user_prompt: Option<String>,
    pending_assistant_output: String,
    status: ReplStatus,
    scroll: u16,
    max_scroll: u16,
    busy: bool,
    runtime_ready: bool,
}

struct ReplStatus {
    left: String,
    right: String,
}

impl ReplApp {
    fn new(cli: &CliOptions) -> Self {
        let mut messages = Vec::new();
        messages.push(ReplLine {
            kind: ReplLineKind::System,
            text: "Entering repl mode. Type /help for commands.".to_string(),
        });
        if cli.debug {
            for line in collect_debug_banner_lines(cli) {
                messages.push(ReplLine {
                    kind: ReplLineKind::Debug,
                    text: line,
                });
            }
        }
        Self {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            messages,
            chat_history: Vec::new(),
            pending_user_prompt: None,
            pending_assistant_output: String::new(),
            status: ReplStatus {
                left: "Loading model...".to_string(),
                right: String::new(),
            },
            scroll: 0,
            max_scroll: 0,
            busy: false,
            runtime_ready: false,
        }
    }

    fn push_line(&mut self, kind: ReplLineKind, text: impl Into<String>) {
        self.messages.push(ReplLine {
            kind,
            text: text.into(),
        });
        self.scroll_to_bottom();
    }

    fn append_chunk(&mut self, kind: ReplLineKind, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        if let Some(last) = self.messages.last_mut() {
            if last.kind == kind {
                last.text.push_str(chunk);
                self.scroll_to_bottom();
                return;
            }
        }
        self.push_line(kind, chunk.to_string());
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll = self.max_scroll;
    }

    fn set_status(&mut self, text: impl Into<String>) {
        self.status.left = text.into();
        self.status.right.clear();
    }

    fn set_progress(&mut self, progress: RuntimeProgress) {
        let phase_label = match progress.phase {
            RuntimePhase::Prefill => "prefill",
            RuntimePhase::Decode => "decode",
            RuntimePhase::Ready => "ready",
        };
        let second_label = match progress.phase {
            RuntimePhase::Decode => "decode",
            _ => "decode",
        };
        let left = match progress.phase {
            RuntimePhase::Ready => "Ready".to_string(),
            RuntimePhase::Prefill | RuntimePhase::Decode => format!(
                "{phase_label} {} tok | {second_label} {} tok",
                progress.prefill_tokens, progress.decode_tokens
            ),
        };
        let right = if progress.context_limit == 0 {
            String::new()
        } else {
            let mut parts = Vec::new();
            if let Some(tok_s) = progress.tokens_per_second {
                parts.push(format!("{tok_s:.1} tok/s"));
            }
            parts.push(format!(
                "ctx {}/{}",
                progress.context_used, progress.context_limit
            ));
            parts.join(" | ")
        };
        self.status = ReplStatus { left, right };
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_index = None;
    }

    fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self
            .input
            .char_indices()
            .take_while(|(idx, _)| *idx < self.cursor)
            .map(|(idx, _)| idx)
            .last()
            .unwrap_or(0);
        self.input.drain(prev..self.cursor);
        self.cursor = prev;
    }

    fn delete(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let next = self
            .input
            .char_indices()
            .find(|(idx, _)| *idx > self.cursor)
            .map(|(idx, _)| idx)
            .unwrap_or(self.input.len());
        self.input.drain(self.cursor..next);
    }

    fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor = self
            .input
            .char_indices()
            .take_while(|(idx, _)| *idx < self.cursor)
            .map(|(idx, _)| idx)
            .last()
            .unwrap_or(0);
    }

    fn move_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        self.cursor = self
            .input
            .char_indices()
            .find(|(idx, _)| *idx > self.cursor)
            .map(|(idx, _)| idx)
            .unwrap_or(self.input.len());
    }

    fn apply_tab_completion(&mut self) {
        let completed = expand_repl_tab_completion(&self.input);
        self.input = completed;
        self.cursor = self.input.len();
    }

    fn push_history(&mut self, entry: &str) {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.history.last().map(|last| last.as_str()) == Some(trimmed) {
            return;
        }
        self.history.push(trimmed.to_string());
        self.history_index = None;
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next_index = match self.history_index {
            Some(0) => 0,
            Some(idx) => idx.saturating_sub(1),
            None => self.history.len().saturating_sub(1),
        };
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(idx) = self.history_index else {
            return;
        };
        if idx + 1 >= self.history.len() {
            self.history_index = None;
            self.input.clear();
            self.cursor = 0;
            return;
        }
        let next_index = idx + 1;
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.cursor = self.input.len();
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn new() -> Result<Self, String> {
        enable_raw_mode().map_err(|e| format!("failed to enable raw mode: {e}"))?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| format!("failed to enter alternate screen: {e}"))?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal =
            Terminal::new(backend).map_err(|e| format!("failed to init terminal: {e}"))?;
        terminal
            .clear()
            .map_err(|e| format!("failed to clear terminal: {e}"))?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

enum WorkerCommand {
    RunPrompt {
        prompt: String,
        chat_history: Vec<ChatMessage>,
    },
    Shutdown,
}

enum WorkerEvent {
    RuntimeReady(Result<(), String>),
    Runtime(RuntimeEvent),
    TurnFinished(Result<(), String>),
}

pub(crate) fn run(cli: &CliOptions) -> Result<(), String> {
    let mut terminal = TerminalGuard::new()?;
    let mut app = ReplApp::new(cli);
    let (command_tx, event_rx) = spawn_worker(cli.clone());
    let mut startup_prompt = if cli.prompt.trim().is_empty() {
        None
    } else {
        Some(cli.prompt.trim().to_string())
    };

    loop {
        drain_worker_events(&mut app, &event_rx)?;
        if app.runtime_ready && !app.busy {
            if let Some(prompt) = startup_prompt.take() {
                if dispatch_prompt(&mut app, &command_tx, cli, &prompt)? {
                    break;
                }
            }
        }

        terminal
            .terminal
            .draw(|frame| draw(frame, &mut app))
            .map_err(|e| format!("terminal draw failed: {e}"))?;

        let cursor_x = 1 + app.input[..min(app.cursor, app.input.len())]
            .chars()
            .count() as u16;
        let size = terminal
            .terminal
            .size()
            .map_err(|e| format!("terminal size failed: {e}"))?;
        let input_y = size.height.saturating_sub(3);
        terminal
            .terminal
            .set_cursor_position((cursor_x, input_y))
            .map_err(|e| format!("cursor position failed: {e}"))?;

        if !event::poll(Duration::from_millis(50)).map_err(|e| format!("event poll failed: {e}"))? {
            continue;
        }
        let Event::Key(key) = event::read().map_err(|e| format!("event read failed: {e}"))? else {
            continue;
        };
        if handle_key_event(&mut app, &command_tx, cli, key)? {
            break;
        }
    }

    let _ = command_tx.send(WorkerCommand::Shutdown);
    Ok(())
}

fn spawn_worker(cli: CliOptions) -> (Sender<WorkerCommand>, Receiver<WorkerEvent>) {
    let (command_tx, command_rx) = mpsc::channel::<WorkerCommand>();
    let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
    thread::spawn(move || worker_main(cli, command_rx, event_tx));
    (command_tx, event_rx)
}

fn worker_main(
    cli: CliOptions,
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
) {
    let callback_tx = event_tx.clone();
    let callback: RuntimeEventCallback = std::sync::Arc::new(move |event| {
        let _ = callback_tx.send(WorkerEvent::Runtime(event));
    });

    match ModelRuntime::load_for_repl(&cli) {
        Ok(mut runtime) => {
            runtime.set_debug_mode(cli.debug);
            let _ = event_tx.send(WorkerEvent::RuntimeReady(Ok(())));
            while let Ok(command) = command_rx.recv() {
                match command {
                    WorkerCommand::RunPrompt {
                        prompt,
                        chat_history,
                    } => {
                        runtime.set_runtime_event_callback(Some(callback.clone()));
                        let result =
                            run_worker_turn(&mut runtime, &cli, &prompt, &chat_history, &callback);
                        runtime.set_runtime_event_callback(None);
                        let _ = event_tx.send(WorkerEvent::TurnFinished(result));
                    }
                    WorkerCommand::Shutdown => break,
                }
            }
        }
        Err(err) => {
            let _ = event_tx.send(WorkerEvent::RuntimeReady(Err(err)));
        }
    }
}

fn run_worker_turn(
    runtime: &mut ModelRuntime,
    cli: &CliOptions,
    prompt: &str,
    chat_history: &[ChatMessage],
    callback: &RuntimeEventCallback,
) -> Result<(), String> {
    if cli.tools_enabled {
        agent::run_agent_loop_collect_with_history_callback(
            runtime,
            cli,
            chat_history,
            prompt,
            Some(callback),
        )?;
        return Ok(());
    }

    let mut messages = chat_history.to_vec();
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: prompt.to_string(),
    });
    let output = runtime.generate_chat_messages_for_repl(&messages, &cli.system_prompt)?;
    if output.trim().is_empty() {
        emit_runtime_event(
            Some(callback),
            RuntimeEvent::Info("<empty response>".to_string()),
        );
    }
    Ok(())
}

fn drain_worker_events(app: &mut ReplApp, event_rx: &Receiver<WorkerEvent>) -> Result<(), String> {
    loop {
        match event_rx.try_recv() {
            Ok(WorkerEvent::RuntimeReady(Ok(()))) => {
                app.runtime_ready = true;
                app.set_status("Ready");
                app.push_line(ReplLineKind::System, "Runtime ready.");
            }
            Ok(WorkerEvent::RuntimeReady(Err(err))) => {
                app.runtime_ready = false;
                app.busy = false;
                app.set_status("Load failed");
                app.push_line(ReplLineKind::Error, format!("Runtime load failed: {err}"));
            }
            Ok(WorkerEvent::Runtime(event)) => match event {
                RuntimeEvent::Output(text) => {
                    app.pending_assistant_output.push_str(&text);
                    app.append_chunk(ReplLineKind::Assistant, &text);
                }
                RuntimeEvent::Debug(text) => app.push_line(ReplLineKind::Debug, text),
                RuntimeEvent::Info(text) => app.push_line(ReplLineKind::System, text),
                RuntimeEvent::Error(text) => app.push_line(ReplLineKind::Error, text),
                RuntimeEvent::Progress(progress) => app.set_progress(progress),
            },
            Ok(WorkerEvent::TurnFinished(Ok(()))) => {
                app.busy = false;
                if let Some(prompt) = app.pending_user_prompt.take() {
                    app.chat_history.push(ChatMessage {
                        role: ChatRole::User,
                        content: prompt,
                    });
                    let assistant = app.pending_assistant_output.trim().to_string();
                    if !assistant.is_empty() {
                        app.chat_history.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: assistant,
                        });
                    }
                }
                app.pending_assistant_output.clear();
                if app.runtime_ready && app.status.left == "Running model..." {
                    app.set_status("Ready");
                }
            }
            Ok(WorkerEvent::TurnFinished(Err(err))) => {
                app.busy = false;
                app.pending_user_prompt = None;
                app.pending_assistant_output.clear();
                app.set_status("Error");
                app.push_line(ReplLineKind::Error, format!("Turn failed: {err}"));
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return Err("repl worker disconnected unexpectedly".to_string());
            }
        }
    }
    Ok(())
}

fn handle_key_event(
    app: &mut ReplApp,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    key: KeyEvent,
) -> Result<bool, String> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Esc => return Ok(true),
        KeyCode::Enter => {
            let submitted = app.input.trim().to_string();
            if submitted.is_empty() {
                return Ok(false);
            }
            app.push_history(&submitted);
            app.clear_input();
            if dispatch_prompt(app, command_tx, cli, &submitted)? {
                return Ok(true);
            }
        }
        KeyCode::Tab => app.apply_tab_completion(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete(),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Home => app.cursor = 0,
        KeyCode::End => app.cursor = app.input.len(),
        KeyCode::Up => app.history_up(),
        KeyCode::Down => app.history_down(),
        KeyCode::PageUp => app.scroll = app.scroll.saturating_sub(5),
        KeyCode::PageDown => app.scroll = app.scroll.saturating_add(5).min(app.max_scroll),
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => app.insert_char(ch),
        _ => {}
    }
    Ok(false)
}

fn dispatch_prompt(
    app: &mut ReplApp,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    input: &str,
) -> Result<bool, String> {
    app.push_line(ReplLineKind::User, format!("> {input}"));
    match handle_repl_command(cli, input) {
        crate::app::ReplCommandAction::Exit => return Ok(true),
        crate::app::ReplCommandAction::Messages(lines) => {
            for line in lines {
                app.push_line(ReplLineKind::System, line);
            }
            if !app.busy && app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ModelPrompt(prompt) => {
            if !app.runtime_ready {
                app.push_line(ReplLineKind::Error, "Runtime is not ready yet.");
                return Ok(false);
            }
            if app.busy {
                app.push_line(
                    ReplLineKind::System,
                    "A turn is already running. Wait for it to finish before submitting another prompt.",
                );
                return Ok(false);
            }
            app.busy = true;
            app.set_status("Running model...");
            app.pending_user_prompt = Some(prompt.clone());
            app.pending_assistant_output.clear();
            command_tx
                .send(WorkerCommand::RunPrompt {
                    prompt,
                    chat_history: app.chat_history.clone(),
                })
                .map_err(|e| format!("failed to send prompt to repl worker: {e}"))?;
        }
    }
    Ok(false)
}

fn draw(frame: &mut Frame<'_>, app: &mut ReplApp) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let transcript_lines = app
        .messages
        .iter()
        .flat_map(render_message_lines)
        .collect::<Vec<_>>();
    let visible_rows = layout[0].height.saturating_sub(2) as usize;
    let max_scroll = transcript_lines.len().saturating_sub(visible_rows) as u16;
    app.max_scroll = max_scroll;
    app.scroll = app.scroll.min(app.max_scroll);
    let transcript = Paragraph::new(transcript_lines)
        .block(Block::default().borders(Borders::ALL).title("Transcript"))
        .wrap(Wrap { trim: false })
        .scroll((app.scroll, 0));
    frame.render_widget(transcript, layout[0]);

    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title("Input"))
        .wrap(Wrap { trim: false });
    frame.render_widget(input, layout[1]);

    let status_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(
                app.status
                    .right
                    .chars()
                    .count()
                    .saturating_add(1)
                    .min(layout[2].width.saturating_sub(1) as usize) as u16,
            ),
        ])
        .split(layout[2]);
    let status_left = Paragraph::new(app.status.left.as_str())
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: false });
    frame.render_widget(status_left, status_layout[0]);
    let status_right = Paragraph::new(app.status.right.as_str())
        .alignment(Alignment::Right)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: false });
    frame.render_widget(status_right, status_layout[1]);
}

fn render_message_lines(msg: &ReplLine) -> Vec<Line<'static>> {
    let (prefix, style) = match msg.kind {
        ReplLineKind::System => (
            "sys",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        ReplLineKind::User => (
            "you",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        ReplLineKind::Assistant => (
            "llm",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        ReplLineKind::Debug => (
            "dbg",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        ReplLineKind::Error => (
            "err",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
    };
    msg.text
        .lines()
        .map(|line| {
            Line::from(vec![
                Span::styled(format!("[{prefix}] "), style),
                Span::raw(line.to_string()),
            ])
        })
        .collect()
}
