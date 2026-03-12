use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) enum RuntimePhase {
    Prefill,
    Decode,
    Ready,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeProgress {
    pub(crate) phase: RuntimePhase,
    pub(crate) prefill_tokens: usize,
    pub(crate) decode_tokens: usize,
    pub(crate) tokens_per_second: Option<f64>,
    pub(crate) context_used: usize,
    pub(crate) context_limit: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimeEvent {
    Output(String),
    Debug(String),
    Info(String),
    Error(String),
    Progress(RuntimeProgress),
}

pub(crate) type RuntimeEventCallback = Arc<dyn Fn(RuntimeEvent) + Send + Sync + 'static>;

pub(crate) fn emit_runtime_event(callback: Option<&RuntimeEventCallback>, event: RuntimeEvent) {
    if let Some(callback) = callback {
        callback(event);
    }
}
