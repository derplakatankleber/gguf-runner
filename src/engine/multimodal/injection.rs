use crate::engine::types::{EncodedPrompt, PlaceholderSpan};
use std::collections::HashMap;

pub(crate) type PrefillEmbeddingMap = HashMap<usize, Vec<f32>>;

#[derive(Clone, Debug)]
pub(crate) struct ImageEmbeddingSequence {
    pub(crate) tokens: Vec<Vec<f32>>,
}

fn validate_image_spans(encoded: &EncodedPrompt) -> Result<(), String> {
    let mut prev_end = 0usize;
    for span in &encoded.image_spans {
        if span.token_len < 2 {
            return Err(format!(
                "image placeholder span[{}] is too short: token_len={} (expected at least 2 for image begin/end markers)",
                span.media_index, span.token_len
            ));
        }
        if span.token_start < prev_end {
            return Err(format!(
                "image placeholder span[{}] overlaps with previous span",
                span.media_index
            ));
        }
        prev_end = span.token_start + span.token_len;
    }
    Ok(())
}

fn image_marker_tokens(encoded: &EncodedPrompt, span: &PlaceholderSpan) -> (i32, i32, i32) {
    let span_tokens = &encoded.token_ids[span.token_start..span.token_start + span.token_len];
    let begin = span_tokens[0];
    let end = span_tokens[span_tokens.len() - 1];
    let placeholder = if span.token_len >= 3 {
        span_tokens[1]
    } else {
        begin
    };
    (begin, placeholder, end)
}

pub(crate) fn expand_prompt_with_image_embeddings(
    encoded: &EncodedPrompt,
    image_embeddings: &[ImageEmbeddingSequence],
    expected_embedding_dim: usize,
) -> Result<(Vec<i32>, PrefillEmbeddingMap), String> {
    validate_image_spans(encoded)?;
    if encoded.image_spans.len() != image_embeddings.len() {
        return Err(format!(
            "image embedding expansion mismatch: {} prompt image span(s) but {} embedding group(s)",
            encoded.image_spans.len(),
            image_embeddings.len()
        ));
    }

    let mut out_tokens: Vec<i32> = Vec::new();
    let mut injected_embeddings: PrefillEmbeddingMap = HashMap::new();
    let mut src_cursor = 0usize;

    for (image_idx, span) in encoded.image_spans.iter().enumerate() {
        if span.token_start + span.token_len > encoded.token_ids.len() {
            return Err(format!(
                "image placeholder span[{}] exceeds prompt token range",
                span.media_index
            ));
        }
        if src_cursor > span.token_start {
            return Err(format!(
                "internal image span traversal error at span[{}]",
                span.media_index
            ));
        }

        out_tokens.extend_from_slice(&encoded.token_ids[src_cursor..span.token_start]);

        let (image_begin, image_placeholder, image_end) = image_marker_tokens(encoded, span);
        out_tokens.push(image_begin);

        let seq = &image_embeddings[image_idx];
        if seq.tokens.is_empty() {
            return Err(format!(
                "image embedding sequence[{}] is empty; at least one embedding token is required",
                image_idx
            ));
        }
        for (tok_idx, emb) in seq.tokens.iter().enumerate() {
            if emb.len() != expected_embedding_dim {
                return Err(format!(
                    "image embedding dim mismatch for image {} token {}: got {}, expected {}",
                    image_idx,
                    tok_idx,
                    emb.len(),
                    expected_embedding_dim
                ));
            }
            let dst_pos = out_tokens.len();
            out_tokens.push(image_placeholder);
            injected_embeddings.insert(dst_pos, emb.clone());
        }

        out_tokens.push(image_end);
        src_cursor = span.token_start + span.token_len;
    }

    out_tokens.extend_from_slice(&encoded.token_ids[src_cursor..]);
    Ok((out_tokens, injected_embeddings))
}
