use crate::cli::CliOptions;
use crate::engine::io::{get_gguf_string_from_map, parse_gguf_file};
use crate::engine::kernels::{argmax, sample, softmax, TopKSampler};
use crate::engine::multimodal::{
    build_vision_encoder_from_mmproj, expand_prompt_with_image_embeddings,
    validate_mmproj_for_backend, VisionEncoder,
};
use crate::engine::profiling::{prof_end, prof_start, record_forward_pass, PROF_TRANSFORMER_NS};
use crate::engine::types::{
    Config, ContentPart, EncodedPrompt, GGUFFile, GenerationRequest, MediaRef, MultimodalBackend,
    MultimodalWeights, PlaceholderSpan, ThinkMode, Tokenizer, TransformerWeights, XorShiftRng,
};
use crate::engine::vision::{
    load_audio_chunk_samples, load_video_chunk_tensors, prepare_audios_for_multimodal,
    prepare_images_for_multimodal, prepare_videos_for_multimodal, ImageNormalization,
    ImagePreprocessProfile, ImageResizeMode,
};
use image::{ImageFormat, ImageReader};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn time_in_ms() -> i64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() * 1000 + (now.subsec_nanos() as u64 / 1_000_000)) as i64
}

fn repeated_cycle_period(tokens: &[i32]) -> Option<usize> {
    // Detect degenerate decode loops by checking whether the current token suffix
    // reappears several times in a recent lookback window (alignment-agnostic).
    const WINDOWS: &[usize] = &[8, 12, 16, 24, 32, 48, 64, 96, 128];
    for &window in WINDOWS {
        let need = window * 3;
        if tokens.len() < need {
            continue;
        }
        let n = tokens.len();
        let suffix = &tokens[n - window..n];
        let search_start = n.saturating_sub(window * 6);
        let search_end = n - window;
        let mut hits = 0usize;
        for start in search_start..=search_end {
            if &tokens[start..start + window] == suffix {
                hits += 1;
                if hits >= 2 {
                    return Some(window);
                }
            }
        }
    }
    None
}

fn repeated_long_line(output: &str) -> Option<(String, usize)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for line in output.lines().rev().take(96) {
        let normalized = line.trim();
        if normalized.len() < 24 {
            continue;
        }
        if normalized.chars().all(|c| c.is_ascii_punctuation()) {
            continue;
        }
        let entry = counts.entry(normalized.to_string()).or_insert(0);
        *entry += 1;
        if *entry >= 2 {
            return Some((normalized.to_string(), *entry));
        }
    }
    None
}

fn repeated_text_suffix_bytes(output: &str) -> Option<usize> {
    let bytes = output.as_bytes();
    const LENGTHS: &[usize] = &[64, 96, 128, 160, 192, 256];
    for &len in LENGTHS {
        if bytes.len() < len * 2 {
            continue;
        }
        let n = bytes.len();
        let a = &bytes[n - len..n];
        let b = &bytes[n - 2 * len..n - len];
        if a == b {
            return Some(len);
        }
    }
    None
}

const THINK_CLOSE_TAG: &str = "</think>";

fn append_visible_text(
    decoded: &str,
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
) {
    if decoded.is_empty() {
        return;
    }
    let decoded = if output.is_empty() {
        decoded.trim_start_matches(['\n', '\r'])
    } else {
        decoded
    };
    if decoded.is_empty() {
        return;
    }
    if decoded == "\n" {
        *pending_newline = true;
        return;
    }
    if *pending_newline {
        if !output.is_empty() {
            output.push('\n');
            if stream_stdout {
                println!();
            }
        }
        *pending_newline = false;
    }
    output.push_str(decoded);
    if stream_stdout {
        print!("{decoded}");
        let _ = io::stdout().flush();
    }
}

fn decode_utf8_streaming(pending: &mut Vec<u8>, piece: &[u8]) -> String {
    pending.extend_from_slice(piece);
    let mut out = String::new();

    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                out.push_str(valid);
                pending.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    let valid = std::str::from_utf8(&pending[..valid_up_to]).unwrap_or_default();
                    out.push_str(valid);
                    pending.drain(..valid_up_to);
                }
                if err.error_len().is_none() {
                    // Incomplete UTF-8 sequence at the end; wait for more token bytes.
                    break;
                }
                // Invalid UTF-8 sequence: emit replacement and advance one byte.
                out.push('\u{FFFD}');
                if !pending.is_empty() {
                    pending.drain(..1);
                } else {
                    break;
                }
            }
        }
    }

    out
}

fn flush_utf8_pending_lossy(pending: &mut Vec<u8>) -> String {
    if pending.is_empty() {
        return String::new();
    }
    let s = String::from_utf8_lossy(pending).to_string();
    pending.clear();
    s
}

fn has_post_think_response_text(output: &str) -> bool {
    if let Some(idx) = output.rfind(THINK_CLOSE_TAG) {
        let rest = &output[idx + THINK_CLOSE_TAG.len()..];
        return !rest.trim().is_empty();
    }
    false
}

fn split_at_char_boundary(s: &str, byte_idx: usize) -> (&str, &str) {
    let mut i = byte_idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    s.split_at(i)
}

fn hidden_strip_visible_chunk(decoded: &str, tail: &mut String) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut combined = String::new();
    if !tail.is_empty() {
        combined.push_str(tail);
        tail.clear();
    }
    combined.push_str(decoded);

    let mut keep_len = 0usize;
    for tag in [OPEN, CLOSE] {
        let max_prefix = tag.len().saturating_sub(1);
        for k in 1..=max_prefix {
            if combined.ends_with(&tag[..k]) {
                keep_len = keep_len.max(k);
            }
        }
    }
    let split_at = combined.len().saturating_sub(keep_len);
    let (emit_part, tail_part) = split_at_char_boundary(&combined, split_at);
    tail.push_str(tail_part);
    emit_part.replace(OPEN, "").replace(CLOSE, "")
}

fn hidden_finalize_tail(tail: &mut String) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if tail.is_empty() {
        return String::new();
    }
    let raw = std::mem::take(tail);
    if OPEN.starts_with(&raw) || CLOSE.starts_with(&raw) {
        return String::new();
    }
    raw.replace(OPEN, "").replace(CLOSE, "")
}

fn trim_leading_line_breaks_for_first_visible<'a>(text: &'a str, output: &str) -> &'a str {
    if output.is_empty() {
        text.trim_start_matches(['\n', '\r'])
    } else {
        text
    }
}

#[allow(clippy::too_many_arguments)]
fn process_decoded_with_think(
    decoded: &str,
    parse_think_tags: bool,
    think_mode: ThinkMode,
    is_thinking: &mut bool,
    think_tail: &mut String,
    hidden_visible_tail: &mut String,
    output: &mut String,
    pending_newline: &mut bool,
    stream_stdout: bool,
) {
    if decoded.is_empty() {
        return;
    }

    if think_mode == ThinkMode::No {
        if parse_think_tags {
            let cleaned = hidden_strip_visible_chunk(decoded, hidden_visible_tail);
            let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
            append_visible_text(cleaned, output, pending_newline, stream_stdout);
        } else {
            append_visible_text(decoded, output, pending_newline, stream_stdout);
        }
        return;
    }
    if !parse_think_tags {
        append_visible_text(decoded, output, pending_newline, stream_stdout);
        return;
    }

    let mut combined = String::new();
    if !think_tail.is_empty() {
        combined.push_str(think_tail);
        think_tail.clear();
    }
    combined.push_str(decoded);

    if !*is_thinking {
        if think_mode == ThinkMode::Hidden {
            let cleaned = hidden_strip_visible_chunk(&combined, hidden_visible_tail);
            let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
            append_visible_text(cleaned, output, pending_newline, stream_stdout);
        } else {
            append_visible_text(&combined, output, pending_newline, stream_stdout);
        }
        return;
    }

    if let Some(close_idx) = combined.find(THINK_CLOSE_TAG) {
        if think_mode == ThinkMode::Yes {
            let end = close_idx + THINK_CLOSE_TAG.len();
            append_visible_text(&combined[..end], output, pending_newline, stream_stdout);
        } else {
            *pending_newline = false;
        }
        *is_thinking = false;
        let rest = &combined[close_idx + THINK_CLOSE_TAG.len()..];
        if !rest.is_empty() {
            if think_mode == ThinkMode::Hidden || think_mode == ThinkMode::No {
                let cleaned = hidden_strip_visible_chunk(rest, hidden_visible_tail);
                let cleaned = trim_leading_line_breaks_for_first_visible(&cleaned, output);
                append_visible_text(cleaned, output, pending_newline, stream_stdout);
            } else {
                append_visible_text(rest, output, pending_newline, stream_stdout);
            }
        }
        return;
    }

    let keep = THINK_CLOSE_TAG.len().saturating_sub(1);
    if think_mode == ThinkMode::Yes {
        if combined.len() > keep {
            let emit_len = combined.len() - keep;
            let (emit_part, tail_part) = split_at_char_boundary(&combined, emit_len);
            append_visible_text(emit_part, output, pending_newline, stream_stdout);
            think_tail.push_str(tail_part);
        } else {
            think_tail.push_str(&combined);
        }
    } else {
        *pending_newline = false;
        if combined.len() > keep {
            let split_at = combined.len().saturating_sub(keep);
            let (_, tail_part) = split_at_char_boundary(&combined, split_at);
            think_tail.push_str(tail_part);
        } else {
            think_tail.push_str(&combined);
        }
    }
}

pub(crate) struct GenerationSettings {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) repeat_penalty: f32,
    pub(crate) repeat_last_n: usize,
    pub(crate) max_tokens: usize,
    pub(crate) profiling_mode: bool,
    pub(crate) show_tokens: bool,
    pub(crate) debug_mode: bool,
    pub(crate) think_mode: ThinkMode,
    pub(crate) vendor_decode_policy: crate::vendors::VendorDecodePolicy,
    pub(crate) vendor_multimodal_policy: crate::vendors::VendorMultimodalPolicy,
}

#[derive(Clone, Debug)]
struct MmprojSidecarProbe {
    path: String,
    has_vision_encoder: bool,
    has_vision_projector: bool,
    has_audio_encoder: bool,
    n_tensors: u64,
}

pub(crate) struct ModelRuntime {
    gguf: GGUFFile,
    config: Config,
    tokenizer: Tokenizer,
    weights: TransformerWeights,
    settings: GenerationSettings,
    multimodal_weights: Option<MultimodalWeights>,
    mmproj_sidecar: Option<MmprojSidecarProbe>,
    mmproj_candidates: Vec<String>,
    vision_encoder: Option<VisionEncoder>,
    kv_cache_format_logged: bool,
}

impl ModelRuntime {
    const DEFAULT_VIDEO_SAMPLED_FPS: u32 = 1;
    const MAX_VIDEO_DECODED_FRAMES: usize = 3600;
    const VIDEO_CHUNK_SIZE_FRAMES: usize = 32;
    const AUDIO_TARGET_SAMPLE_RATE: u32 = 16_000;
    const AUDIO_MAX_SAMPLES: usize = 16_000 * 3600;
    const AUDIO_CHUNK_SIZE_SAMPLES: usize = 16_000 * 30;
    const VISION_ENCODER_TENSOR_PREFIXES: &'static [&'static str] = &[
        "v.",
        "vision.",
        "visual.",
        "vision_tower.",
        "vision_encoder.",
        "model.vision.",
        "model.visual.",
        "model.vision_tower.",
    ];
    const VISION_PROJECTOR_TENSOR_PREFIXES: &'static [&'static str] = &[
        "mm.",
        "mmproj.",
        "multi_modal_projector.",
        "projector.",
        "model.mmproj.",
        "model.projector.",
        "vision_language_adapter.",
        "model.vision_language_adapter.",
    ];
    const AUDIO_TENSOR_PREFIXES: &'static [&'static str] =
        &["audio.", "aud.", "speech.", "whisper.", "model.audio."];

    fn model_architecture(&self) -> &str {
        get_gguf_string_from_map(&self.gguf.kv, "general.architecture").unwrap_or("unknown")
    }

    fn gguf_has_tensor_with_any_prefix(gguf: &GGUFFile, prefixes: &[&str]) -> bool {
        gguf.tensors.iter().any(|tensor| {
            prefixes
                .iter()
                .any(|prefix| tensor.name.starts_with(prefix))
        })
    }

    fn has_vocab_token(&self, token: &str) -> bool {
        self.gguf.vocab_tokens.iter().any(|entry| entry == token)
    }

    fn has_tensor_with_any_prefix(&self, prefixes: &[&str]) -> bool {
        Self::gguf_has_tensor_with_any_prefix(&self.gguf, prefixes)
    }

    fn has_image_tokens(&self) -> bool {
        self.has_vocab_token("<|vision_start|>")
            && self.has_vocab_token("<|vision_end|>")
            && self.has_vocab_token("<|image_pad|>")
    }

    fn has_video_tokens(&self) -> bool {
        self.has_vocab_token("<|vision_start|>")
            && self.has_vocab_token("<|vision_end|>")
            && self.has_vocab_token("<|video_pad|>")
    }

    fn has_audio_tokens(&self) -> bool {
        self.has_vocab_token("<|audio_pad|>")
    }

    fn supports_external_vision(&self) -> bool {
        self.mmproj_sidecar
            .as_ref()
            .map(|probe| probe.has_vision_encoder && probe.has_vision_projector)
            .unwrap_or(false)
    }

    fn supports_external_audio(&self) -> bool {
        self.mmproj_sidecar
            .as_ref()
            .map(|probe| probe.has_audio_encoder)
            .unwrap_or(false)
    }

    fn effective_supports_image(&self) -> bool {
        self.config.capabilities.supports_native_image
            || (self.has_image_tokens() && self.supports_external_vision())
    }

    fn effective_supports_video(&self) -> bool {
        self.config.capabilities.supports_native_video
            || (self.has_video_tokens() && self.supports_external_vision())
    }

    fn effective_supports_audio(&self) -> bool {
        self.config.capabilities.supports_native_audio
            || (self.has_audio_tokens() && self.supports_external_audio())
    }

    fn mmproj_summary(&self) -> String {
        if let Some(probe) = &self.mmproj_sidecar {
            format!(
                "mmproj(path='{}', n_tensors={}, vision_encoder={}, vision_projector={}, audio={})",
                probe.path,
                probe.n_tensors,
                probe.has_vision_encoder,
                probe.has_vision_projector,
                probe.has_audio_encoder
            )
        } else if self.mmproj_candidates.is_empty() {
            "mmproj(path=not-searched)".to_string()
        } else {
            format!(
                "mmproj(path=not-found, searched=[{}])",
                self.mmproj_candidates.join(", ")
            )
        }
    }

    fn discover_mmproj_candidates(model_path: &str) -> Vec<PathBuf> {
        fn push_unique(candidates: &mut Vec<PathBuf>, path: PathBuf) {
            if !candidates.iter().any(|existing| existing == &path) {
                candidates.push(path);
            }
        }

        let model = Path::new(model_path);
        let parent = model.parent().unwrap_or_else(|| Path::new("."));
        let file_name = model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let stem = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let mut candidates: Vec<PathBuf> = Vec::new();
        if !file_name.is_empty() {
            push_unique(&mut candidates, parent.join(format!("mmproj-{file_name}")));
        }
        if !stem.is_empty() {
            push_unique(&mut candidates, parent.join(format!("mmproj-{stem}.gguf")));
            push_unique(&mut candidates, parent.join(format!("{stem}.mmproj.gguf")));
        }
        push_unique(&mut candidates, parent.join("mmproj.gguf"));

        if let Ok(entries) = fs::read_dir(parent) {
            let mut discovered: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .map(|name| {
                            let lowered = name.to_ascii_lowercase();
                            lowered.starts_with("mmproj") && lowered.ends_with(".gguf")
                        })
                        .unwrap_or(false)
                })
                .collect();
            discovered.sort();
            for path in discovered {
                push_unique(&mut candidates, path);
            }
        }

        candidates
    }

    fn strip_quant_suffix(stem: &str) -> String {
        let mut kept: Vec<&str> = Vec::new();
        for token in stem.split('-') {
            let t = token.trim().to_ascii_lowercase();
            let q_prefix_is_quant = t
                .strip_prefix('q')
                .and_then(|rest| rest.chars().next())
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            let is_quant_token = q_prefix_is_quant || t == "f16" || t == "f32" || t == "bf16";
            if is_quant_token {
                break;
            }
            kept.push(token);
        }
        if kept.is_empty() {
            stem.to_string()
        } else {
            kept.join("-")
        }
    }

    fn normalize_alnum_lower(input: &str) -> String {
        input
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    }

    fn split_alnum_tokens(input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut cur = String::new();
        for ch in input.chars() {
            if ch.is_ascii_alphanumeric() {
                cur.push(ch.to_ascii_lowercase());
            } else if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            tokens.push(cur);
        }
        tokens
    }

    fn is_size_token(token: &str) -> bool {
        token
            .strip_suffix('b')
            .map(|prefix| !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    }

    fn is_active_size_token(token: &str) -> bool {
        token
            .strip_prefix('a')
            .and_then(|t| t.strip_suffix('b'))
            .map(|middle| !middle.is_empty() && middle.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    }

    fn required_model_variant_tokens(checkpoint: &str) -> Vec<String> {
        let stem = Path::new(checkpoint)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let mut required = Vec::new();
        for token in Self::split_alnum_tokens(stem) {
            if Self::is_size_token(&token) || Self::is_active_size_token(&token) {
                required.push(token);
            }
        }
        required
    }

    fn sidecar_descriptor_tokens(sidecar_path: &str, sidecar: &GGUFFile) -> HashSet<String> {
        let mut combined = String::new();
        combined.push_str(sidecar_path);
        combined.push(' ');
        for key in [
            "general.name",
            "general.basename",
            "general.finetune",
            "general.base_model.0.name",
            "general.base_model.0.repo_url",
            "general.repo_url",
        ] {
            if let Some(value) = get_gguf_string_from_map(&sidecar.kv, key) {
                combined.push_str(value);
                combined.push(' ');
            }
        }
        Self::split_alnum_tokens(&combined).into_iter().collect()
    }

    fn validate_mmproj_variant_match(
        checkpoint: &str,
        sidecar_path: &str,
        sidecar: &GGUFFile,
    ) -> Result<(), String> {
        let required_tokens = Self::required_model_variant_tokens(checkpoint);
        if required_tokens.is_empty() {
            return Ok(());
        }
        let sidecar_tokens = Self::sidecar_descriptor_tokens(sidecar_path, sidecar);
        for token in required_tokens {
            if !sidecar_tokens.contains(&token) {
                return Err(format!(
                    "mmproj/model variant mismatch: model requires token '{token}' (derived from checkpoint name), but sidecar metadata/name does not contain it"
                ));
            }
        }
        Ok(())
    }

    fn score_mmproj_candidate(
        path: &Path,
        normalized_model_key: &str,
        backend: MultimodalBackend,
        policy: crate::vendors::VendorMultimodalPolicy,
    ) -> i32 {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let normalized_file = Self::normalize_alnum_lower(file_name);
        let mut score = 0i32;
        if !normalized_model_key.is_empty() && normalized_file.contains(normalized_model_key) {
            score += 1_000;
        }
        for hint in policy.mmproj_filename_score_hints {
            if normalized_file.contains(hint.token) {
                if backend == hint.backend {
                    score += hint.match_score;
                } else {
                    score += hint.mismatch_score;
                }
            }
        }
        if normalized_file.contains("mmproj") {
            score += 10;
        }
        score
    }

    fn probe_mmproj_sidecar(
        checkpoint: &str,
        cfg: &Config,
        debug_mode: bool,
    ) -> Result<(Option<MmprojSidecarProbe>, Vec<String>), String> {
        let candidates = Self::discover_mmproj_candidates(checkpoint);
        let candidate_strings = candidates
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let model_stem = Path::new(checkpoint)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let model_key = Self::normalize_alnum_lower(&Self::strip_quant_suffix(
            model_stem.to_lowercase().as_str(),
        ));
        let multimodal_policy = crate::vendors::multimodal_policy(cfg);
        let mut existing_candidates = candidates
            .into_iter()
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        existing_candidates.sort_by(|a, b| {
            let sa = Self::score_mmproj_candidate(
                a,
                &model_key,
                cfg.capabilities.multimodal_backend,
                multimodal_policy,
            );
            let sb = Self::score_mmproj_candidate(
                b,
                &model_key,
                cfg.capabilities.multimodal_backend,
                multimodal_policy,
            );
            sb.cmp(&sa)
                .then_with(|| a.to_string_lossy().cmp(&b.to_string_lossy()))
        });
        for path in existing_candidates {
            let sidecar_path = path.to_string_lossy().into_owned();
            let sidecar = match parse_gguf_file(&sidecar_path, debug_mode) {
                Ok(sidecar) => sidecar,
                Err(e) => {
                    if debug_mode {
                        eprintln!(
                            "Skipping mmproj sidecar '{}': failed to parse GGUF ({e})",
                            sidecar_path
                        );
                    }
                    continue;
                }
            };
            let probe = MmprojSidecarProbe {
                path: sidecar_path.clone(),
                has_vision_encoder: Self::gguf_has_tensor_with_any_prefix(
                    &sidecar,
                    Self::VISION_ENCODER_TENSOR_PREFIXES,
                ),
                has_vision_projector: Self::gguf_has_tensor_with_any_prefix(
                    &sidecar,
                    Self::VISION_PROJECTOR_TENSOR_PREFIXES,
                ),
                has_audio_encoder: Self::gguf_has_tensor_with_any_prefix(
                    &sidecar,
                    Self::AUDIO_TENSOR_PREFIXES,
                ),
                n_tensors: sidecar.n_tensors,
            };
            if let Err(e) = validate_mmproj_for_backend(cfg, &sidecar) {
                if debug_mode {
                    eprintln!(
                        "Skipping mmproj sidecar '{}': not compatible with backend '{}' ({e})",
                        sidecar_path,
                        cfg.capabilities.multimodal_backend.as_str()
                    );
                }
                continue;
            }
            if let Err(e) = Self::validate_mmproj_variant_match(checkpoint, &sidecar_path, &sidecar)
            {
                if debug_mode {
                    eprintln!(
                        "Skipping mmproj sidecar '{}': checkpoint-variant mismatch ({e})",
                        sidecar_path
                    );
                }
                continue;
            }
            return Ok((Some(probe), candidate_strings));
        }
        Ok((None, candidate_strings))
    }

    fn summarize_tensor_prefixes(&self, max_items: usize) -> String {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for tensor in &self.gguf.tensors {
            let prefix = tensor
                .name
                .split('.')
                .next()
                .unwrap_or("unknown")
                .to_string();
            *counts.entry(prefix).or_insert(0) += 1;
        }
        let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries
            .into_iter()
            .take(max_items)
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn native_media_probe_details(&self) -> String {
        let has_vision_start = self.has_vocab_token("<|vision_start|>");
        let has_vision_end = self.has_vocab_token("<|vision_end|>");
        let has_image_pad = self.has_vocab_token("<|image_pad|>");
        let has_video_pad = self.has_vocab_token("<|video_pad|>");
        let has_audio_pad = self.has_vocab_token("<|audio_pad|>");
        let has_vision_encoder =
            self.has_tensor_with_any_prefix(Self::VISION_ENCODER_TENSOR_PREFIXES);
        let has_vision_projector =
            self.has_tensor_with_any_prefix(Self::VISION_PROJECTOR_TENSOR_PREFIXES);
        let has_audio_encoder = self.has_tensor_with_any_prefix(Self::AUDIO_TENSOR_PREFIXES);
        let top_prefixes = self.summarize_tensor_prefixes(8);
        let vision_placeholders =
            has_vision_start && has_vision_end && (has_image_pad || has_video_pad);
        let likely_text_only_export =
            vision_placeholders && !has_vision_encoder && !has_vision_projector;
        let text_only_hint = if likely_text_only_export {
            " hint: chat-template vision placeholders exist, but no vision/projector tensor groups were found in this GGUF; this artifact is likely text-only."
        } else {
            ""
        };
        let sidecar_hint = if self.mmproj_sidecar.is_none()
            && !self
                .settings
                .vendor_multimodal_policy
                .missing_sidecar_hint
                .is_empty()
        {
            self.settings.vendor_multimodal_policy.missing_sidecar_hint
        } else {
            ""
        };

        format!(
            "probe details: arch='{}', n_tensors={}, tokens(vision_start={} vision_end={} image_pad={} video_pad={} audio_pad={}), tensor_groups_main(vision_encoder={} vision_projector={} audio={}), effective_support(image={} video={} audio={}), {}, top_tensor_prefixes=[{}].{}{}",
            self.model_architecture(),
            self.gguf.n_tensors,
            has_vision_start,
            has_vision_end,
            has_image_pad,
            has_video_pad,
            has_audio_pad,
            has_vision_encoder,
            has_vision_projector,
            has_audio_encoder,
            self.effective_supports_image(),
            self.effective_supports_video(),
            self.effective_supports_audio(),
            self.mmproj_summary(),
            top_prefixes,
            text_only_hint,
            sidecar_hint,
        )
    }

    fn ensure_native_media_support(
        &self,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) -> Result<(), String> {
        if image_count == 0 && video_count == 0 && audio_count == 0 {
            return Ok(());
        }

        let capabilities = self.config.capabilities;
        if capabilities.multimodal_backend == MultimodalBackend::None {
            return Err(format!(
                "media inputs require a native multimodal backend, but model architecture '{}' is text-only in this runner",
                self.model_architecture()
            ));
        }

        let backend = capabilities.multimodal_backend.as_str();
        if image_count > 0 && !self.effective_supports_image() {
            return Err(format!(
                "image inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports image=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }
        if video_count > 0 && !self.effective_supports_video() {
            return Err(format!(
                "video inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports video=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }
        if audio_count > 0 && !self.effective_supports_audio() {
            return Err(format!(
                "audio inputs require native multimodal tensors/components for backend '{backend}', but capability probe reports audio=false for this GGUF (image={} video={} audio={}). {}",
                capabilities.supports_native_image,
                capabilities.supports_native_video,
                capabilities.supports_native_audio,
                self.native_media_probe_details(),
            ));
        }

        Ok(())
    }

    fn image_preprocess_profile(&self) -> ImagePreprocessProfile {
        let fallback_norm = ImageNormalization::MeanStd {
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.2757771],
        };
        if let Some(encoder) = &self.vision_encoder {
            let (mean, std) = encoder.recommended_image_normalization();
            let clip_norm = ImageNormalization::MeanStd { mean, std };
            let base_size = encoder.recommended_image_size().max(224);
            let align_to = encoder.recommended_image_alignment().max(1);
            if self.config.capabilities.multimodal_backend == MultimodalBackend::Qwen35 {
                // Scale image resolution with model embedding dim. At dim=2048 (2B) this
                // yields ~2/3 of the mmproj base_size; at dim=3072 (7B) the full base_size;
                // beyond that the resolution continues to grow for OCR and fine-detail tasks,
                // capped at 2× base_size where bilinear position-embedding interpolation still
                // produces reliable results.
                let balanced_size = ((base_size as f32 * self.config.dim as f32 / 3072.0) as usize)
                    .clamp(align_to.max(224), base_size * 2);
                let aligned = if align_to > 1 {
                    (balanced_size / align_to) * align_to
                } else {
                    balanced_size
                };
                let target = aligned.max(align_to).max(224);
                return ImagePreprocessProfile::new_with_mode(
                    target,
                    target,
                    clip_norm,
                    ImageResizeMode::FitWithin,
                    align_to,
                );
            }
            return ImagePreprocessProfile::new_with_mode(
                base_size,
                base_size,
                clip_norm,
                ImageResizeMode::CenterCrop,
                align_to,
            );
        }

        match self.config.capabilities.multimodal_backend {
            MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
                ImagePreprocessProfile::new(224, 224, fallback_norm)
            }
            MultimodalBackend::None => {
                ImagePreprocessProfile::new(448, 448, ImageNormalization::UnitRange)
            }
        }
    }

    fn create_center_square_crop_file(
        image_path: &str,
        temp_file_prefix: &str,
    ) -> Result<Option<String>, String> {
        let reader = ImageReader::open(image_path)
            .map_err(|e| format!("cannot open image '{image_path}' for detail crop: {e}"))?;
        let decoded = reader
            .decode()
            .map_err(|e| format!("cannot decode image '{image_path}' for detail crop: {e}"))?;
        let rgb = decoded.to_rgb8();
        let src_w = rgb.width() as usize;
        let src_h = rgb.height() as usize;
        if src_w == 0 || src_h == 0 {
            return Ok(None);
        }
        if src_w == src_h {
            return Ok(None);
        }

        let side = src_w.min(src_h);
        if side < 64 {
            return Ok(None);
        }
        let crop_x = (src_w - side) / 2;
        let crop_y = (src_h - side) / 2;
        let cropped =
            image::imageops::crop_imm(&rgb, crop_x as u32, crop_y as u32, side as u32, side as u32)
                .to_image();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let file_prefix = if temp_file_prefix.is_empty() {
            "gguf-runner-detail"
        } else {
            temp_file_prefix
        };
        let file_name = format!("{file_prefix}-{}-{now}.png", std::process::id());
        let out_path = std::env::temp_dir().join(file_name);
        cropped
            .save_with_format(&out_path, ImageFormat::Png)
            .map_err(|e| {
                format!(
                    "cannot save detail crop for image '{image_path}' to '{}': {e}",
                    out_path.display()
                )
            })?;
        Ok(Some(out_path.to_string_lossy().into_owned()))
    }

    fn expand_request_for_vendor_detail_crop(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationRequest, String> {
        let detail_crop_policy = self.settings.vendor_multimodal_policy.detail_crop;
        if !detail_crop_policy.enabled {
            return Ok(request.clone());
        }
        if self.config.n_layers > detail_crop_policy.max_layers {
            // Keep large variants on single-image path unless explicitly requested otherwise.
            return Ok(request.clone());
        }
        if request
            .parts
            .iter()
            .any(|part| matches!(part, ContentPart::Video(_)))
            || request
                .parts
                .iter()
                .any(|part| matches!(part, ContentPart::Audio(_)))
        {
            return Ok(request.clone());
        }

        let mut image_indices: Vec<(usize, String)> = Vec::new();
        for (idx, part) in request.parts.iter().enumerate() {
            if let ContentPart::Image(img) = part {
                image_indices.push((idx, img.path.clone()));
            }
        }
        if image_indices.len() != 1 {
            return Ok(request.clone());
        }
        let (image_part_idx, image_path) = &image_indices[0];
        let Some(crop_path) =
            Self::create_center_square_crop_file(image_path, detail_crop_policy.temp_file_prefix)?
        else {
            return Ok(request.clone());
        };

        let mut parts: Vec<ContentPart> = Vec::with_capacity(request.parts.len() + 2);
        for (idx, part) in request.parts.iter().enumerate() {
            parts.push(part.clone());
            if idx == *image_part_idx {
                if !detail_crop_policy.note_text.is_empty() {
                    parts.push(ContentPart::Text(detail_crop_policy.note_text.to_string()));
                }
                parts.push(ContentPart::Image(MediaRef {
                    path: crop_path.clone(),
                }));
            }
        }

        Ok(GenerationRequest {
            system_prompt: request.system_prompt.clone(),
            parts,
        })
    }

    fn validate_placeholder_spans(
        &self,
        spans: &[PlaceholderSpan],
        label: &str,
    ) -> Result<(), String> {
        let mut prev_end = 0usize;
        for span in spans {
            if span.token_len == 0 {
                return Err(format!(
                    "prompt placeholder span for {label}[{}] has zero token length",
                    span.media_index
                ));
            }
            if span.token_start < prev_end {
                return Err(format!(
                    "prompt placeholder spans for {label} overlap or are out of order around media index {}",
                    span.media_index
                ));
            }
            prev_end = span.token_start.saturating_add(span.token_len);
        }
        Ok(())
    }

    fn validate_encoded_prompt_media_alignment(
        &self,
        encoded: &EncodedPrompt,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) -> Result<(), String> {
        self.validate_placeholder_spans(&encoded.image_spans, "image")?;
        self.validate_placeholder_spans(&encoded.video_spans, "video")?;
        self.validate_placeholder_spans(&encoded.audio_spans, "audio")?;

        if encoded.image_spans.len() != image_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} image placeholder span(s), but request contains {image_count} image input(s)",
                encoded.image_spans.len()
            ));
        }
        if encoded.video_spans.len() != video_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} video placeholder span(s), but request contains {video_count} video input(s)",
                encoded.video_spans.len()
            ));
        }
        if encoded.audio_spans.len() != audio_count {
            return Err(format!(
                "prompt/media mismatch: encoded {} audio placeholder span(s), but request contains {audio_count} audio input(s)",
                encoded.audio_spans.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn load(cli: &CliOptions) -> Result<Self, String> {
        let mut max_tokens = cli.max_tokens;
        let debug_mode = cli.debug;
        let checkpoint = &cli.model;
        if debug_mode {
            eprintln!("Loading GGUF model: {checkpoint}");
            eprintln!(
                "Sampling: temperature={}, top_k={}, top_p={}, repeat_penalty={}, repeat_last_n={}",
                cli.temperature, cli.top_k, cli.top_p, cli.repeat_penalty, cli.repeat_last_n
            );
        }

        let gguf = parse_gguf_file(checkpoint, debug_mode)?;

        if debug_mode {
            eprintln!(
                "GGUF metadata: version={}, tensors={}, kv={}, tensor_data_start={} bytes",
                gguf.version, gguf.n_tensors, gguf.n_kv, gguf.tensor_data_start
            );
        }

        let mut config = crate::vendors::build_config_from_gguf(&gguf, debug_mode)?;
        let tokenizer_policy = crate::vendors::tokenizer_policy(&config);
        let vendor_multimodal_policy = crate::vendors::multimodal_policy(&config);
        let vendor_runtime_debug_policy = crate::vendors::runtime_debug_policy(&config);
        let mut tokenizer = crate::engine::tokenizer::init_tokenizer_from_gguf(
            &gguf,
            &mut config,
            tokenizer_policy,
            debug_mode,
        )?;
        tokenizer.use_sentencepiece = config.is_gemma3;
        let media_requested =
            !cli.images.is_empty() || !cli.videos.is_empty() || !cli.audios.is_empty();
        let (mmproj_sidecar, mmproj_candidates) = if media_requested
            && config.capabilities.multimodal_backend != MultimodalBackend::None
        {
            Self::probe_mmproj_sidecar(checkpoint, &config, debug_mode)?
        } else {
            (None, Vec::new())
        };
        if debug_mode
            && media_requested
            && config.capabilities.multimodal_backend != MultimodalBackend::None
        {
            if let Some(probe) = &mmproj_sidecar {
                eprintln!(
                    "Detected llama-style mmproj sidecar: path='{}', tensors={}, vision_encoder={}, vision_projector={}, audio={}",
                    probe.path,
                    probe.n_tensors,
                    probe.has_vision_encoder,
                    probe.has_vision_projector,
                    probe.has_audio_encoder
                );
            } else if !mmproj_candidates.is_empty() {
                eprintln!(
                    "No llama-style mmproj sidecar found. searched candidates: [{}]",
                    mmproj_candidates.join(", ")
                );
            }
        }

        let vision_encoder = if media_requested {
            if let Some(probe) = &mmproj_sidecar {
                let mmproj = parse_gguf_file(&probe.path, debug_mode).map_err(|e| {
                    format!(
                        "failed to load llama-style mmproj sidecar '{}' for multimodal backend initialization: {e}",
                        probe.path
                    )
                })?;
                build_vision_encoder_from_mmproj(&config, mmproj)?
            } else {
                None
            }
        } else {
            None
        };

        crate::engine::runtime::apply_context_size_overrides(
            &mut config,
            cli.context_size,
            debug_mode,
        );
        if cli.context_size == 0 && debug_mode {
            if let Some(label) = vendor_runtime_debug_policy.native_context_label {
                eprintln!(
                    "Using {label} native context length {} (model may require a large workspace)",
                    config.seq_len
                );
            }
        }
        if max_tokens == 0 || max_tokens > config.seq_len {
            max_tokens = config.seq_len;
        }

        if let Some(n_threads) = cli.threads {
            crate::engine::runtime::configure_rayon_threads(n_threads, debug_mode);
        }

        let weights = crate::engine::weights::init_weights_from_gguf(&gguf, &config, debug_mode)?;
        let multimodal_weights =
            crate::engine::weights::init_multimodal_weights_from_gguf(&gguf, &config, debug_mode)?;
        let vendor_decode_policy = crate::vendors::decode_policy(&config);
        let settings = GenerationSettings {
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            repeat_penalty: cli.repeat_penalty,
            repeat_last_n: cli.repeat_last_n,
            max_tokens,
            profiling_mode: cli.profiling,
            show_tokens: cli.show_tokens,
            debug_mode,
            think_mode: cli.think_mode,
            vendor_decode_policy,
            vendor_multimodal_policy,
        };

        Ok(Self {
            gguf,
            config,
            tokenizer,
            weights,
            settings,
            multimodal_weights,
            mmproj_sidecar,
            mmproj_candidates,
            vision_encoder,
            kv_cache_format_logged: false,
        })
    }

    pub(crate) fn generate_text(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        stream_stdout: bool,
    ) -> Result<String, String> {
        self.generate_text_with_images(prompt, system_prompt, &[], stream_stdout)
    }

    pub(crate) fn generate_request(
        &mut self,
        request: &GenerationRequest,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let effective_request = self.expand_request_for_vendor_detail_crop(request)?;
        let mut prompt_parts: Vec<&str> = Vec::new();
        let mut images: Vec<String> = Vec::new();
        let mut videos: Vec<String> = Vec::new();
        let mut audios: Vec<String> = Vec::new();

        for part in &effective_request.parts {
            match part {
                ContentPart::Text(text) => prompt_parts.push(text),
                ContentPart::Image(image) => images.push(image.path.clone()),
                ContentPart::Video(video) => videos.push(video.path.clone()),
                ContentPart::Audio(audio) => audios.push(audio.path.clone()),
            }
        }

        let prompt = prompt_parts.join("\n");
        if prompt.trim().is_empty() && images.is_empty() && videos.is_empty() && audios.is_empty() {
            return Err("generation request has no text content".to_string());
        }

        let image_profile = self.image_preprocess_profile();
        self.ensure_native_media_support(images.len(), videos.len(), audios.len())
            .map_err(|e| {
                format!("native multimodal execution required (fallback disabled): {e}")
            })?;
        if (!images.is_empty() || !videos.is_empty() || !audios.is_empty())
            && self.multimodal_weights.is_none()
        {
            return Err(format!(
                "native media path selected but multimodal weights for backend '{}' were not initialized",
                self.config.capabilities.multimodal_backend.as_str()
            ));
        }

        let encoded_prompt = crate::vendors::encode_generation_request(
            &mut self.tokenizer,
            &self.config,
            &effective_request,
            self.settings.think_mode,
        );
        self.validate_encoded_prompt_media_alignment(
            &encoded_prompt,
            images.len(),
            videos.len(),
            audios.len(),
        )?;
        if self.settings.debug_mode {
            eprintln!(
                "Encoded prompt: tokens={}, image_spans={}, video_spans={}, audio_spans={}",
                encoded_prompt.token_ids.len(),
                encoded_prompt.image_spans.len(),
                encoded_prompt.video_spans.len(),
                encoded_prompt.audio_spans.len()
            );
        }

        let mut preprocess_summary: Vec<String> = Vec::new();
        let mut prepared_images = Vec::new();

        if !images.is_empty() {
            prepared_images = prepare_images_for_multimodal(&images, image_profile)?;
            if self.settings.debug_mode {
                let first = &prepared_images[0];
                eprintln!(
                    "Prepared {} image tensor(s); first image: path='{}', width={}, height={}, elements={}",
                    prepared_images.len(),
                    first.path,
                    first.width,
                    first.height,
                    first.element_count()
                );
            }
            preprocess_summary.push(format!("images={}", prepared_images.len()));
        }
        if !videos.is_empty() {
            let prepared = prepare_videos_for_multimodal(
                &videos,
                image_profile,
                Self::DEFAULT_VIDEO_SAMPLED_FPS,
                Self::MAX_VIDEO_DECODED_FRAMES,
                Self::VIDEO_CHUNK_SIZE_FRAMES,
            )?;
            if self.settings.debug_mode {
                let first = &prepared[0];
                let (chunk_start, chunk_frames, decoded_chunk_frames) =
                    if let Some(chunk0) = first.chunks.first() {
                        let decoded = load_video_chunk_tensors(first, 0, image_profile)?;
                        (chunk0.start_frame, chunk0.frame_paths.len(), decoded.len())
                    } else {
                        (0, 0, 0)
                    };
                eprintln!(
                    "Prepared {} video tensor(s); first video: path='{}', fps={}, size={}x{}, frames={}, chunks={}, first_chunk_start={}, first_chunk_frames={}, first_chunk_decoded={}",
                    prepared.len(),
                    first.path,
                    first.sampled_fps,
                    first.frame_width,
                    first.frame_height,
                    first.frame_count,
                    first.chunks.len(),
                    chunk_start,
                    chunk_frames,
                    decoded_chunk_frames
                );
            }
            preprocess_summary.push(format!("videos={}", prepared.len()));
        }
        if !audios.is_empty() {
            let prepared = prepare_audios_for_multimodal(
                &audios,
                Self::AUDIO_TARGET_SAMPLE_RATE,
                Self::AUDIO_MAX_SAMPLES,
                Self::AUDIO_CHUNK_SIZE_SAMPLES,
            )?;
            if self.settings.debug_mode {
                let first = &prepared[0];
                let first_chunk_samples = if !first.chunks.is_empty() {
                    load_audio_chunk_samples(first, 0)?.len()
                } else {
                    0
                };
                eprintln!(
                    "Prepared {} audio tensor(s); first audio: path='{}', sample_rate={}, channels={}, samples={}, chunks={}, first_chunk_samples={}",
                    prepared.len(),
                    first.path,
                    first.sample_rate,
                    first.channels,
                    first.total_samples,
                    first.chunks.len(),
                    first_chunk_samples
                );
            }
            preprocess_summary.push(format!("audios={}", prepared.len()));
        }

        let mut prefill_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut prompt_tokens = encoded_prompt.token_ids.clone();

        if !prepared_images.is_empty() {
            let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
                let mmproj_note = self
                    .mmproj_sidecar
                    .as_ref()
                    .map(|probe| format!(" (llama-style mmproj sidecar loaded: '{}')", probe.path))
                    .unwrap_or_default();
                format!(
                    "native image preprocessing succeeded ({}), but no compatible native vision encoder is initialized for backend '{}'{}",
                    preprocess_summary.join(", "),
                    self.config.capabilities.multimodal_backend.as_str(),
                    mmproj_note
                )
            })?;
            let image_embeddings = encoder.encode_images(&prepared_images)?;
            let (expanded_tokens, injected) = expand_prompt_with_image_embeddings(
                &encoded_prompt,
                &image_embeddings,
                self.config.input_embedding_dim,
            )?;
            prompt_tokens = expanded_tokens;
            prefill_embeddings = injected;
        }

        if !videos.is_empty() || !audios.is_empty() {
            return Err(format!(
                "native media preprocessing completed ({}), but video/audio embedding execution is not implemented yet",
                preprocess_summary.join(", ")
            ));
        }

        if self.settings.debug_mode {
            if let Some(mm) = &self.multimodal_weights {
                eprintln!(
                    "Multimodal weights ready: backend={}, total_tensors={}, vision={}, projector={}, audio={}",
                    mm.backend.as_str(),
                    mm.total_tensor_count(),
                    mm.vision_tensor_names.len(),
                    mm.projector_tensor_names.len(),
                    mm.audio_tensor_names.len()
                );
            }
            eprintln!(
                "Prepared multimodal prefill: tokens={} injected_embeddings={} images={} videos={} audios={}",
                prompt_tokens.len(),
                prefill_embeddings.len(),
                encoded_prompt.image_spans.len(),
                encoded_prompt.video_spans.len(),
                encoded_prompt.audio_spans.len()
            );
        }

        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token);
        }
        if prompt_tokens.len() > self.config.seq_len {
            prompt_tokens.truncate(self.config.seq_len);
            prefill_embeddings.retain(|k, _| *k < self.config.seq_len);
        }

        self.generate_from_prefill(prompt_tokens, prefill_embeddings, stream_stdout)
    }

    fn generate_from_prefill(
        &mut self,
        prompt_tokens: Vec<i32>,
        prefill_injected_embeddings: HashMap<usize, Vec<f32>>,
        stream_stdout: bool,
    ) -> Result<String, String> {
        let hidden_retry_enabled = self.settings.think_mode == ThinkMode::Hidden;
        let retry_prompt_tokens = if hidden_retry_enabled {
            Some(prompt_tokens.clone())
        } else {
            None
        };
        let retry_prefill_embeddings = if hidden_retry_enabled {
            Some(prefill_injected_embeddings.clone())
        } else {
            None
        };

        let temperature = self.settings.temperature;
        let top_k = self.settings.top_k;
        let top_p = self.settings.top_p;
        let repetition_penalty = self.settings.repeat_penalty;
        let repeat_last_n = self.settings.repeat_last_n;
        let max_new_tokens = self.settings.max_tokens;
        let profiling_mode = self.settings.profiling_mode;
        let show_tokens = self.settings.show_tokens;
        let debug_mode = self.settings.debug_mode;

        let mut token = prompt_tokens[0];
        let mut next: i32;
        let mut pos = 0usize;
        let mut start = 0i64;

        let mut state = crate::engine::runtime::malloc_run_state(&self.config)?;
        if debug_mode && !self.kv_cache_format_logged {
            eprintln!("KV cache format: {:?}", state.kv_cache_format);
            self.kv_cache_format_logged = true;
        }
        let mut rng = XorShiftRng::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        let mut topk_sampler = TopKSampler::new();
        let mut warned_top_p_without_top_k = false;

        let use_repetition_penalty = repetition_penalty != 1.0 && repeat_last_n > 0;
        let mut recent_tokens = if use_repetition_penalty {
            VecDeque::with_capacity(repeat_last_n)
        } else {
            VecDeque::new()
        };
        let mut unique_recent_tokens = if use_repetition_penalty {
            HashSet::with_capacity(repeat_last_n)
        } else {
            HashSet::new()
        };
        let mut pending_newline = false;
        let mut output = String::new();
        let mut generated_tokens = Vec::new();
        let mut utf8_pending: Vec<u8> = Vec::new();
        let mut think_tail = String::new();
        let mut hidden_visible_tail = String::new();
        let mut hidden_think_token_count = 0usize;

        // Think mode state: track whether we're currently inside a <think>...</think> block.
        // The prompt already ends with "<think>\n" for Yes/Hidden modes, so generation starts
        // inside the thinking block. For No mode the prompt closes it immediately.
        let think_mode = self.settings.think_mode;
        let decode_policy = self.settings.vendor_decode_policy;
        let thinking_active = decode_policy.parse_think_tags && think_mode != ThinkMode::No;
        let mut is_thinking = thinking_active;
        if thinking_active && think_mode == ThinkMode::Yes && stream_stdout {
            println!("<think>");
            let _ = io::stdout().flush();
        }
        let stop_token_ids = decode_policy
            .stop_token_literals
            .iter()
            .filter_map(|literal| self.tokenizer.find_special_token(literal))
            .collect::<Vec<_>>();

        let total_limit = prompt_tokens
            .len()
            .saturating_add(max_new_tokens)
            .min(self.config.seq_len);
        let hidden_mode_caps = if decode_policy.parse_think_tags && think_mode == ThinkMode::Hidden
        {
            let new_budget = total_limit.saturating_sub(prompt_tokens.len());
            let vendor_base = if self.config.is_qwen35 || self.config.is_qwen3next {
                384usize
            } else if self.config.is_qwen3vl {
                320usize
            } else {
                256usize
            };
            let layer_mult = if self.config.n_layers >= 64 {
                3usize
            } else if self.config.n_layers >= 32 {
                2usize
            } else {
                1usize
            };
            let ctx_mult = if self.config.seq_len >= 262_144 {
                2usize
            } else {
                1usize
            };
            let hard_think_cap_max = if self.config.n_layers >= 48 || self.config.seq_len >= 262_144
            {
                1536usize
            } else {
                1024usize
            };
            let hard_total_cap_max = hard_think_cap_max.saturating_mul(4);
            let mut think_cap = vendor_base
                .saturating_mul(layer_mult)
                .saturating_mul(ctx_mult);
            if new_budget > 0 {
                let max_cap = hard_think_cap_max.min(new_budget.max(96));
                think_cap = think_cap.clamp(96, max_cap);
            } else {
                think_cap = 64;
            }
            let mut total_cap = think_cap.saturating_mul(4).min(hard_total_cap_max);
            if new_budget > 0 {
                total_cap = total_cap.min(new_budget);
                let min_total = (think_cap + 64).min(new_budget).max(think_cap + 1);
                total_cap = total_cap.max(min_total);
            } else {
                total_cap = think_cap + 64;
            }
            Some((think_cap, total_cap))
        } else {
            None
        };
        if let Some((think_cap, total_cap)) = hidden_mode_caps {
            if debug_mode {
                eprintln!(
                    "Hidden-think policy: think_cap={} total_cap={} (n_layers={} seq_len={} budget={})",
                    think_cap,
                    total_cap,
                    self.config.n_layers,
                    self.config.seq_len,
                    total_limit.saturating_sub(prompt_tokens.len())
                );
            }
        }
        while pos < total_limit {
            if token < 0 || token as usize >= self.config.vocab_size {
                return Err(format!("token id out of bounds: {token}"));
            }

            let prof_t0 = prof_start();
            if let Some(embedding) = prefill_injected_embeddings.get(&pos) {
                crate::engine::runtime::transformer_with_embedding(
                    embedding,
                    pos,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                )?;
            } else {
                crate::engine::runtime::transformer(
                    token as usize,
                    pos,
                    &self.config,
                    &mut state,
                    &self.weights,
                    self.gguf.mapped.as_slice(),
                )?;
            }
            prof_end(&PROF_TRANSFORMER_NS, prof_t0);
            if profiling_mode {
                record_forward_pass();
            }

            if debug_mode
                && pos >= prompt_tokens.len().saturating_sub(1)
                && pos < prompt_tokens.len() + 3
            {
                let mut top: Vec<(usize, f32)> = state.logits[..self.config.vocab_size]
                    .iter()
                    .copied()
                    .enumerate()
                    .collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                eprint!("[DEBUG pos={pos}] Top 5 logits: ");
                for (id, v) in top.into_iter().take(5) {
                    let decoded = self
                        .tokenizer
                        .decode_token(id as i32)
                        .unwrap_or_else(|| "?".to_string())
                        .replace('\n', "\\n")
                        .replace('\r', "\\r");
                    eprint!("{id}({v:.2},\"{decoded}\") ");
                }
                eprintln!();
            }

            if pos < prompt_tokens.len().saturating_sub(1) {
                next = prompt_tokens[pos + 1];
            } else {
                if use_repetition_penalty {
                    unique_recent_tokens.clear();
                    for &tok in &recent_tokens {
                        unique_recent_tokens.insert(tok);
                    }
                    for tok in unique_recent_tokens.iter().copied() {
                        if tok >= 0 && (tok as usize) < self.config.vocab_size {
                            let idx = tok as usize;
                            if state.logits[idx] > 0.0 {
                                state.logits[idx] /= repetition_penalty;
                            } else {
                                state.logits[idx] *= repetition_penalty;
                            }
                        }
                    }
                }

                if temperature == 0.0 {
                    next = argmax(&state.logits[..self.config.vocab_size]) as i32;
                } else if top_k > 0 {
                    next = topk_sampler.sample_top_k_top_p(
                        &state.logits[..self.config.vocab_size],
                        temperature,
                        top_k,
                        top_p,
                        &mut rng,
                    ) as i32;
                } else {
                    if top_p < 1.0 && debug_mode && !warned_top_p_without_top_k {
                        eprintln!("Note: -top_p is ignored unless -top_k > 0");
                        warned_top_p_without_top_k = true;
                    }
                    for q in 0..self.config.vocab_size {
                        state.logits[q] /= temperature;
                    }
                    softmax(
                        &mut state.logits[..self.config.vocab_size],
                        self.config.vocab_size,
                    );
                    next = sample(&state.logits[..self.config.vocab_size], &mut rng) as i32;
                }

                if use_repetition_penalty {
                    if recent_tokens.len() == repeat_last_n {
                        recent_tokens.pop_front();
                    }
                    recent_tokens.push_back(next);
                }
            }

            if pos >= prompt_tokens.len().saturating_sub(1)
                && next != self.tokenizer.eot_token
                && next != self.tokenizer.eos_token
            {
                if let Some(bytes) = self.tokenizer.decode_token_bytes(next) {
                    let decoded = decode_utf8_streaming(&mut utf8_pending, &bytes);
                    process_decoded_with_think(
                        &decoded,
                        decode_policy.parse_think_tags,
                        think_mode,
                        &mut is_thinking,
                        &mut think_tail,
                        &mut hidden_visible_tail,
                        &mut output,
                        &mut pending_newline,
                        stream_stdout,
                    );
                }
            }

            token = next;
            pos += 1;

            if start == 0 {
                start = time_in_ms();
            }

            if pos >= prompt_tokens.len().saturating_sub(1) {
                generated_tokens.push(token);
                if let Some((hidden_think_cap, hidden_total_cap)) = hidden_mode_caps {
                    if is_thinking {
                        hidden_think_token_count = hidden_think_token_count.saturating_add(1);
                        if hidden_think_token_count >= hidden_think_cap {
                            is_thinking = false;
                            think_tail.clear();
                            pending_newline = false;
                            if debug_mode || show_tokens {
                                eprintln!(
                                    "\nNote: forced exit from hidden thinking block after {} tokens without </think>",
                                    hidden_think_cap
                                );
                            }
                        }
                    }
                    if generated_tokens.len() >= hidden_total_cap {
                        if debug_mode || show_tokens {
                            eprintln!(
                                "\nStopping hidden mode after {} generated tokens to avoid unbounded run",
                                hidden_total_cap
                            );
                        }
                        break;
                    }
                }
                if token == self.tokenizer.eos_token || token == self.tokenizer.eot_token {
                    break;
                }
                if stop_token_ids.contains(&token) {
                    break;
                }
                if self.settings.vendor_decode_policy.deterministic_loop_guard
                    && generated_tokens.len() % 16 == 0
                {
                    if let Some(len) = repeated_text_suffix_bytes(&output) {
                        if debug_mode || show_tokens {
                            eprintln!(
                                "\nStopping due to repeated output suffix block (len={len} bytes)"
                            );
                        }
                        break;
                    }
                    if let Some((line, repeats)) = repeated_long_line(&output) {
                        if debug_mode || show_tokens {
                            eprintln!(
                                "\nStopping due to repeated output line (repeats={repeats}): {line}"
                            );
                        }
                        break;
                    }
                    if temperature == 0.0 {
                        if let Some(period) = repeated_cycle_period(&generated_tokens) {
                            if debug_mode || show_tokens {
                                eprintln!(
                                    "\nStopping due to repeated token cycle (window={period}, repeated suffix)"
                                );
                            }
                            break;
                        }
                    }
                }
            }
        }

        let pending_decoded = flush_utf8_pending_lossy(&mut utf8_pending);
        process_decoded_with_think(
            &pending_decoded,
            decode_policy.parse_think_tags,
            think_mode,
            &mut is_thinking,
            &mut think_tail,
            &mut hidden_visible_tail,
            &mut output,
            &mut pending_newline,
            stream_stdout,
        );
        if !think_tail.is_empty() {
            if think_mode == ThinkMode::Yes {
                append_visible_text(
                    &think_tail,
                    &mut output,
                    &mut pending_newline,
                    stream_stdout,
                );
            }
            think_tail.clear();
        }
        if think_mode == ThinkMode::Hidden || think_mode == ThinkMode::No {
            let trailing = hidden_finalize_tail(&mut hidden_visible_tail);
            let trailing = trim_leading_line_breaks_for_first_visible(&trailing, &output);
            append_visible_text(trailing, &mut output, &mut pending_newline, stream_stdout);
        }
        if decode_policy.parse_think_tags && think_mode == ThinkMode::Yes && is_thinking {
            append_visible_text(
                THINK_CLOSE_TAG,
                &mut output,
                &mut pending_newline,
                stream_stdout,
            );
            if debug_mode || show_tokens {
                eprintln!("\nNote: auto-closed missing </think> at end of generation");
            }
        }
        if decode_policy.parse_think_tags
            && think_mode == ThinkMode::Yes
            && !has_post_think_response_text(&output)
        {
            let mut promoted = output.trim().to_string();
            if let Some(prefix) = promoted.strip_suffix(THINK_CLOSE_TAG) {
                promoted = prefix.trim().to_string();
            }
            if !promoted.is_empty() {
                if stream_stdout {
                    println!();
                    print!("{promoted}");
                    let _ = io::stdout().flush();
                }
                output = promoted;
                if debug_mode || show_tokens {
                    eprintln!("\nNote: promoted think-only content to final response text");
                }
            }
        }

        let end = time_in_ms();
        if (debug_mode || show_tokens) && pos > 1 {
            let elapsed_ms = (end - start).max(1) as f64;
            eprintln!(
                "\nachieved tok/s: {:.3}",
                (pos - 1) as f64 / elapsed_ms * 1000.0
            );
        } else if stream_stdout && !output.is_empty() {
            println!();
        }

        if hidden_retry_enabled && output.trim().is_empty() {
            if debug_mode || show_tokens {
                eprintln!(
                    "\nNote: hidden think mode produced no visible output; retrying with think=no"
                );
            }
            if let (Some(retry_prompt_tokens), Some(retry_prefill_embeddings)) =
                (retry_prompt_tokens, retry_prefill_embeddings)
            {
                let original_think_mode = self.settings.think_mode;
                self.settings.think_mode = ThinkMode::No;
                let retry_result = self.generate_from_prefill(
                    retry_prompt_tokens,
                    retry_prefill_embeddings,
                    stream_stdout,
                );
                self.settings.think_mode = original_think_mode;
                return retry_result;
            }
        }

        Ok(output)
    }

    pub(crate) fn generate_text_with_images(
        &mut self,
        prompt: &str,
        system_prompt: &str,
        images: &[String],
        stream_stdout: bool,
    ) -> Result<String, String> {
        if !images.is_empty() {
            let mut parts = Vec::with_capacity(1 + images.len());
            // Keep image placeholders ahead of text so multimodal templates can bind media spans first.
            for image in images {
                parts.push(ContentPart::Image(crate::engine::types::MediaRef {
                    path: image.clone(),
                }));
            }
            let mut effective_prompt = prompt.to_string();
            if !self
                .settings
                .vendor_multimodal_policy
                .image_prompt_suffix
                .is_empty()
            {
                effective_prompt
                    .push_str(self.settings.vendor_multimodal_policy.image_prompt_suffix);
            }
            if !prompt.trim().is_empty() {
                parts.push(ContentPart::Text(effective_prompt));
            }
            let req = GenerationRequest {
                system_prompt: system_prompt.to_string(),
                parts,
            };
            return self.generate_request(&req, stream_stdout);
        }

        let debug_mode = self.settings.debug_mode;
        let mut prompt_tokens: Vec<i32> = crate::vendors::encode_chat_prompt(
            &mut self.tokenizer,
            &self.config,
            prompt,
            system_prompt,
            images.len(),
            self.settings.think_mode,
        );

        if prompt_tokens.is_empty() {
            prompt_tokens.push(self.tokenizer.bos_token);
        }
        if prompt_tokens.len() > self.config.seq_len {
            prompt_tokens.truncate(self.config.seq_len);
        }
        if debug_mode {
            eprintln!("Prompt tokens: {}", prompt_tokens.len());
            let preview = prompt_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("Prompt token ids: [{preview}]");
        }

        let prefill_injected_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
        self.generate_from_prefill(prompt_tokens, prefill_injected_embeddings, stream_stdout)
    }
}
