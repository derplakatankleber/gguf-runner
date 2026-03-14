use crate::app::agent;
use crate::app::events::{
    emit_runtime_event, RuntimeEvent, RuntimeEventCallback, RuntimePhase, RuntimeProgress,
};
use crate::app::generation::ModelRuntime;
use crate::app::{collect_debug_banner_lines, expand_repl_tab_completion, handle_repl_command};
use crate::cli::CliOptions;
use crate::vendors::{ChatMessage, ChatRole};
use crossterm::cursor::{Hide, MoveTo, RestorePosition, SavePosition, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::style::Print;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size, Clear, ClearType};
use std::cmp::min;
use std::io::{self, Stdout, Write};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

const FOOTER_HEIGHT: u16 = 2;

struct ReplApp {
    input: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    active_images: Vec<String>,
    chat_history: Vec<ChatMessage>,
    pending_user_prompt: Option<String>,
    pending_assistant_output: String,
    assistant_line_open: bool,
    status: ReplStatus,
    busy: bool,
    runtime_ready: bool,
}

struct ReplStatus {
    left: String,
    right: String,
}

impl ReplApp {
    fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            active_images: Vec::new(),
            chat_history: Vec::new(),
            pending_user_prompt: None,
            pending_assistant_output: String::new(),
            assistant_line_open: false,
            status: ReplStatus {
                left: "Loading model...".to_string(),
                right: String::new(),
            },
            busy: false,
            runtime_ready: false,
        }
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
        let left = match progress.phase {
            RuntimePhase::Ready => "Ready".to_string(),
            RuntimePhase::Prefill | RuntimePhase::Decode => format!(
                "{phase_label} {} tok | decode {} tok",
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
                "ctx {}",
                context_gauge(progress.context_used, progress.context_limit, 10)
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
    stdout: Stdout,
    width: u16,
    height: u16,
}

impl TerminalGuard {
    fn new() -> Result<Self, String> {
        enable_raw_mode().map_err(|e| format!("failed to enable raw mode: {e}"))?;
        let mut stdout = io::stdout();
        execute!(stdout, Hide).map_err(|e| format!("failed to hide cursor: {e}"))?;
        writeln!(stdout).map_err(|e| format!("failed to initialize repl footer: {e}"))?;
        writeln!(stdout).map_err(|e| format!("failed to initialize repl footer: {e}"))?;
        stdout
            .flush()
            .map_err(|e| format!("failed to flush terminal: {e}"))?;
        let (width, height) = size().map_err(|e| format!("failed to get terminal size: {e}"))?;
        let mut guard = Self {
            stdout,
            width,
            height,
        };
        guard.configure_scroll_region()?;
        Ok(guard)
    }

    fn refresh_size(&mut self) -> Result<(), String> {
        let (width, height) = size().map_err(|e| format!("failed to get terminal size: {e}"))?;
        self.width = width;
        self.height = height;
        self.configure_scroll_region()
    }

    fn configure_scroll_region(&mut self) -> Result<(), String> {
        let output_bottom = self.output_bottom_row_1based();
        write!(self.stdout, "\x1b[1;{}r", output_bottom)
            .map_err(|e| format!("failed to set scroll region: {e}"))?;
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush scroll region: {e}"))
    }

    fn output_bottom_row_1based(&self) -> u16 {
        self.height.saturating_sub(FOOTER_HEIGHT).max(1)
    }

    fn footer_status_row(&self) -> u16 {
        self.height.saturating_sub(FOOTER_HEIGHT)
    }

    fn footer_input_row(&self) -> u16 {
        self.height.saturating_sub(1)
    }

    fn ensure_output_line_closed(&mut self, app: &mut ReplApp) -> Result<(), String> {
        if app.assistant_line_open {
            execute!(self.stdout, Print("\r\n"))
                .map_err(|e| format!("failed to end assistant line: {e}"))?;
            self.stdout
                .flush()
                .map_err(|e| format!("failed to flush assistant line ending: {e}"))?;
            app.assistant_line_open = false;
        }
        Ok(())
    }

    fn print_prefixed_lines(
        &mut self,
        app: &mut ReplApp,
        prefix: &str,
        text: &str,
    ) -> Result<(), String> {
        self.ensure_output_line_closed(app)?;
        let output_row = self.output_bottom_row_1based().saturating_sub(1);
        for line in text.lines() {
            execute!(
                self.stdout,
                MoveTo(0, output_row),
                Clear(ClearType::CurrentLine),
                Print(prefix),
                Print(line),
                Print("\r\n")
            )
            .map_err(|e| format!("failed to print repl output: {e}"))?;
        }
        if text.is_empty() {
            execute!(
                self.stdout,
                MoveTo(0, output_row),
                Clear(ClearType::CurrentLine),
                Print(prefix),
                Print("\r\n")
            )
            .map_err(|e| format!("failed to print repl output: {e}"))?;
        }
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush repl output: {e}"))
    }

    fn print_assistant_chunk(&mut self, app: &mut ReplApp, chunk: &str) -> Result<(), String> {
        if chunk.is_empty() {
            return Ok(());
        }
        let output_row = self.output_bottom_row_1based().saturating_sub(1);
        for segment in chunk.split_inclusive('\n') {
            if !app.assistant_line_open {
                execute!(
                    self.stdout,
                    MoveTo(0, output_row),
                    Clear(ClearType::CurrentLine),
                    Print("[llm] ")
                )
                .map_err(|e| format!("failed to start assistant line: {e}"))?;
                app.assistant_line_open = true;
            }
            execute!(self.stdout, Print(segment))
                .map_err(|e| format!("failed to stream assistant output: {e}"))?;
            if segment.ends_with('\n') {
                app.assistant_line_open = false;
            }
        }
        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush assistant output: {e}"))
    }

    fn render_footer(&mut self, app: &ReplApp) -> Result<(), String> {
        self.refresh_size()?;
        let preserve_output_cursor = app.busy;
        if preserve_output_cursor {
            execute!(self.stdout, SavePosition, Hide)
                .map_err(|e| format!("failed to save cursor: {e}"))?;
        } else {
            execute!(self.stdout, Show).map_err(|e| format!("failed to show cursor: {e}"))?;
        }

        let status_line = compose_status_line(&app.status, self.width as usize);
        let (input_line, cursor_col) =
            compose_input_line(&app.input, app.cursor, self.width as usize);
        let status_row = self.footer_status_row();
        let input_row = self.footer_input_row();

        execute!(
            self.stdout,
            MoveTo(0, status_row),
            Clear(ClearType::CurrentLine),
            Print(status_line),
            MoveTo(0, input_row),
            Clear(ClearType::CurrentLine),
            Print(input_line)
        )
        .map_err(|e| format!("failed to render repl footer: {e}"))?;

        if preserve_output_cursor {
            execute!(self.stdout, RestorePosition)
                .map_err(|e| format!("failed to restore cursor: {e}"))?;
        } else {
            execute!(self.stdout, MoveTo(cursor_col, input_row))
                .map_err(|e| format!("failed to position input cursor: {e}"))?;
        }

        self.stdout
            .flush()
            .map_err(|e| format!("failed to flush repl footer: {e}"))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let status_row = self.footer_status_row();
        let input_row = self.footer_input_row();
        let _ = write!(self.stdout, "\x1b[r");
        let _ = execute!(
            self.stdout,
            MoveTo(0, status_row),
            Clear(ClearType::CurrentLine),
            MoveTo(0, input_row),
            Clear(ClearType::CurrentLine),
            Show
        );
        let _ = writeln!(self.stdout);
        let _ = self.stdout.flush();
        let _ = disable_raw_mode();
    }
}

enum WorkerCommand {
    RunPrompt {
        prompt: String,
        chat_history: Vec<ChatMessage>,
        active_images: Vec<String>,
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
    let mut app = ReplApp::new();
    let (command_tx, event_rx) = spawn_worker(cli.clone());
    let mut startup_prompt = if cli.prompt.trim().is_empty() {
        None
    } else {
        Some(cli.prompt.trim().to_string())
    };

    terminal.print_prefixed_lines(
        &mut app,
        "[sys] ",
        "Entering repl mode. Type /help for commands.",
    )?;
    if cli.debug {
        for line in collect_debug_banner_lines(cli) {
            terminal.print_prefixed_lines(&mut app, "[dbg] ", &line)?;
        }
    }
    terminal.render_footer(&app)?;

    loop {
        drain_worker_events(&mut app, &mut terminal, &event_rx)?;
        if app.runtime_ready && !app.busy {
            if let Some(prompt) = startup_prompt.take() {
                if dispatch_prompt(&mut app, &mut terminal, &command_tx, cli, &prompt)? {
                    break;
                }
            }
        }

        terminal.render_footer(&app)?;

        if !event::poll(Duration::from_millis(50)).map_err(|e| format!("event poll failed: {e}"))? {
            continue;
        }
        match event::read().map_err(|e| format!("event read failed: {e}"))? {
            Event::Key(key) => {
                if handle_key_event(&mut app, &mut terminal, &command_tx, cli, key)? {
                    break;
                }
            }
            Event::Resize(_, _) => {
                terminal.render_footer(&app)?;
            }
            _ => {}
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
                        active_images,
                    } => {
                        runtime.set_runtime_event_callback(Some(callback.clone()));
                        let result = run_worker_turn(
                            &mut runtime,
                            &cli,
                            &prompt,
                            &chat_history,
                            &active_images,
                            &callback,
                        );
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
    active_images: &[String],
    callback: &RuntimeEventCallback,
) -> Result<(), String> {
    if !active_images.is_empty() {
        if cli.tools_enabled {
            emit_runtime_event(
                Some(callback),
                RuntimeEvent::Info(
                    "Active image context detected; using native multimodal path for this turn."
                        .to_string(),
                ),
            );
        }
        let request =
            build_repl_multimodal_request(prompt, &cli.system_prompt, chat_history, active_images);
        let output = runtime.generate_request(&request, false)?;
        if output.trim().is_empty() {
            emit_runtime_event(
                Some(callback),
                RuntimeEvent::Info("<empty response>".to_string()),
            );
        }
        return Ok(());
    }

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

fn drain_worker_events(
    app: &mut ReplApp,
    terminal: &mut TerminalGuard,
    event_rx: &Receiver<WorkerEvent>,
) -> Result<(), String> {
    loop {
        match event_rx.try_recv() {
            Ok(WorkerEvent::RuntimeReady(Ok(()))) => {
                app.runtime_ready = true;
                app.set_status("Ready");
                terminal.print_prefixed_lines(app, "[sys] ", "Runtime ready.")?;
            }
            Ok(WorkerEvent::RuntimeReady(Err(err))) => {
                app.runtime_ready = false;
                app.busy = false;
                app.set_status("Load failed");
                terminal.print_prefixed_lines(
                    app,
                    "[err] ",
                    &format!("Runtime load failed: {err}"),
                )?;
            }
            Ok(WorkerEvent::Runtime(event)) => match event {
                RuntimeEvent::Output(text) => {
                    app.pending_assistant_output.push_str(&text);
                    terminal.print_assistant_chunk(app, &text)?;
                }
                RuntimeEvent::Debug(text) => terminal.print_prefixed_lines(app, "[dbg] ", &text)?,
                RuntimeEvent::Info(text) => terminal.print_prefixed_lines(app, "[sys] ", &text)?,
                RuntimeEvent::Error(text) => terminal.print_prefixed_lines(app, "[err] ", &text)?,
                RuntimeEvent::Progress(progress) => app.set_progress(progress),
            },
            Ok(WorkerEvent::TurnFinished(Ok(()))) => {
                app.busy = false;
                terminal.ensure_output_line_closed(app)?;
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
                terminal.ensure_output_line_closed(app)?;
                app.set_status("Error");
                terminal.print_prefixed_lines(app, "[err] ", &format!("Turn failed: {err}"))?;
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
    terminal: &mut TerminalGuard,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    key: KeyEvent,
) -> Result<bool, String> {
    if !should_handle_key_event(key) {
        return Ok(false);
    }
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
            if dispatch_prompt(app, terminal, command_tx, cli, &submitted)? {
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
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => app.insert_char(ch),
        _ => {}
    }
    Ok(false)
}

fn should_handle_key_event(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn dispatch_prompt(
    app: &mut ReplApp,
    terminal: &mut TerminalGuard,
    command_tx: &Sender<WorkerCommand>,
    cli: &CliOptions,
    input: &str,
) -> Result<bool, String> {
    terminal.print_prefixed_lines(app, "[you] ", &format!("> {input}"))?;
    match handle_repl_command(cli, input) {
        crate::app::ReplCommandAction::Exit => return Ok(true),
        crate::app::ReplCommandAction::Messages(lines) => {
            for line in lines {
                terminal.print_prefixed_lines(app, "[sys] ", &line)?;
            }
            if !app.busy && app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::AttachImage(path) => {
            let canonical = crate::app::validate_repl_image_path(&path)?;
            if app.active_images.contains(&canonical) {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Image already attached: {canonical}"),
                )?;
            } else {
                app.active_images.push(canonical.clone());
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Attached image: {canonical}"),
                )?;
            }
            if !app.busy && app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ListImages => {
            if app.active_images.is_empty() {
                terminal.print_prefixed_lines(app, "[sys] ", "No active image attachments.")?;
            } else {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    &format!("Active images ({}):", app.active_images.len()),
                )?;
                for image in app.active_images.clone() {
                    terminal.print_prefixed_lines(app, "[sys] ", &format!("  {image}"))?;
                }
            }
            if !app.busy && app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ClearImages => {
            let cleared = app.active_images.len();
            app.active_images.clear();
            terminal.print_prefixed_lines(
                app,
                "[sys] ",
                &format!("Cleared {cleared} active image attachment(s)."),
            )?;
            if !app.busy && app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ClearState => {
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "Cannot clear state while a turn is running.",
                )?;
                return Ok(false);
            }
            let cleared_messages = app.chat_history.len();
            let cleared_images = app.active_images.len();
            app.chat_history.clear();
            app.active_images.clear();
            app.pending_user_prompt = None;
            app.pending_assistant_output.clear();
            app.assistant_line_open = false;
            terminal.print_prefixed_lines(
                app,
                "[sys] ",
                &format!(
                    "Cleared chat state: {} message(s), {} image attachment(s).",
                    cleared_messages, cleared_images
                ),
            )?;
            if app.runtime_ready {
                app.set_status("Ready");
            }
            return Ok(false);
        }
        crate::app::ReplCommandAction::ModelPrompt(prompt) => {
            if !app.runtime_ready {
                terminal.print_prefixed_lines(app, "[err] ", "Runtime is not ready yet.")?;
                return Ok(false);
            }
            if app.busy {
                terminal.print_prefixed_lines(
                    app,
                    "[sys] ",
                    "A turn is already running. Wait for it to finish before submitting another prompt.",
                )?;
                return Ok(false);
            }
            app.busy = true;
            app.set_status("Running model...");
            app.pending_user_prompt = Some(prompt.clone());
            app.pending_assistant_output.clear();
            app.assistant_line_open = false;
            command_tx
                .send(WorkerCommand::RunPrompt {
                    prompt,
                    chat_history: app.chat_history.clone(),
                    active_images: app.active_images.clone(),
                })
                .map_err(|e| format!("failed to send prompt to repl worker: {e}"))?;
        }
    }
    Ok(false)
}

fn build_repl_multimodal_request(
    prompt: &str,
    system_prompt: &str,
    chat_history: &[ChatMessage],
    active_images: &[String],
) -> crate::engine::types::GenerationRequest {
    const MAX_HISTORY_MESSAGES: usize = 12;
    let history_slice = if chat_history.len() > MAX_HISTORY_MESSAGES {
        &chat_history[chat_history.len() - MAX_HISTORY_MESSAGES..]
    } else {
        chat_history
    };

    let mut prompt_text = String::new();
    if !history_slice.is_empty() {
        prompt_text.push_str("Conversation so far:\n");
        for message in history_slice {
            match message.role {
                ChatRole::User => prompt_text.push_str("User: "),
                ChatRole::Assistant => prompt_text.push_str("Assistant: "),
            }
            prompt_text.push_str(&message.content);
            prompt_text.push('\n');
        }
        prompt_text.push('\n');
    }
    prompt_text.push_str(&format!(
        "There are {} active image attachment(s) for this conversation.\nCurrent user message: {}",
        active_images.len(),
        prompt
    ));

    let mut parts = Vec::with_capacity(active_images.len().saturating_add(1));
    for path in active_images {
        parts.push(crate::engine::types::ContentPart::Image(
            crate::engine::types::MediaRef { path: path.clone() },
        ));
    }
    parts.push(crate::engine::types::ContentPart::Text(prompt_text));
    crate::engine::types::GenerationRequest {
        system_prompt: system_prompt.to_string(),
        parts,
    }
}

fn compose_status_line(status: &ReplStatus, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let left = truncate_for_width(&status.left, width);
    if status.right.is_empty() {
        return left;
    }
    let right = truncate_for_width(&status.right, width);
    let left_len = left.chars().count();
    let right_len = right.chars().count();
    if left_len + right_len + 1 > width {
        return truncate_for_width(&format!("{} {}", left, right), width);
    }
    format!(
        "{}{}{}",
        left,
        " ".repeat(width - left_len - right_len),
        right
    )
}

fn compose_input_line(input: &str, cursor: usize, width: usize) -> (String, u16) {
    let prefix = "> ";
    if width <= prefix.len() {
        return (prefix[..width].to_string(), width as u16);
    }
    let available = width - prefix.len();
    let chars = input.chars().collect::<Vec<_>>();
    let cursor_chars = input[..min(cursor, input.len())].chars().count();
    let start = cursor_chars.saturating_sub(available.saturating_sub(1));
    let end = min(chars.len(), start + available);
    let visible = chars[start..end].iter().collect::<String>();
    let mut line = String::with_capacity(width);
    line.push_str(prefix);
    line.push_str(&visible);
    let visible_len = visible.chars().count();
    if prefix.len() + visible_len < width {
        line.push_str(&" ".repeat(width - prefix.len() - visible_len));
    }
    let cursor_col = (prefix.len() + cursor_chars.saturating_sub(start)).min(width);
    (line, cursor_col as u16)
}

fn truncate_for_width(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

fn context_gauge(used: usize, limit: usize, width: usize) -> String {
    if limit == 0 || width == 0 {
        return String::new();
    }
    let used = used.min(limit);
    let filled = ((used * width) + (limit / 2)) / limit;
    let percent = ((used * 100) + (limit / 2)) / limit;
    let filled = filled.min(width);
    format!(
        "[{}{}] {:>3}%",
        "#".repeat(filled),
        ".".repeat(width - filled),
        percent
    )
}

#[cfg(test)]
mod tests {
    use super::should_handle_key_event;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    #[test]
    fn key_event_filter_ignores_release_events() {
        let press = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        let repeat = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Repeat,
            state: KeyEventState::NONE,
        };
        let release = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };

        assert!(should_handle_key_event(press));
        assert!(should_handle_key_event(repeat));
        assert!(!should_handle_key_event(release));
    }
}
