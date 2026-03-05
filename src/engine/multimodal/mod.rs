mod injection;
mod qwen3vl;

use crate::engine::io::get_gguf_string_from_map;
use crate::engine::types::{Config, GGUFFile, MultimodalBackend};
use crate::engine::vision::PreparedImageTensor;
pub(crate) use injection::{expand_prompt_with_image_embeddings, ImageEmbeddingSequence};

pub(crate) enum VisionEncoder {
    Qwen3Vl(qwen3vl::Qwen3VlVisionEncoder),
}

impl VisionEncoder {
    pub(crate) fn recommended_image_size(&self) -> usize {
        match self {
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_size(),
        }
    }

    pub(crate) fn recommended_image_alignment(&self) -> usize {
        match self {
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_alignment(),
        }
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_normalization(),
        }
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<ImageEmbeddingSequence>, String> {
        match self {
            VisionEncoder::Qwen3Vl(enc) => enc.encode_images(images),
        }
    }
}

fn normalize_alpha_num(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
}

fn mmproj_family_markers(mmproj: &GGUFFile) -> (bool, bool) {
    let mut combined = String::new();
    for key in [
        "general.name",
        "general.basename",
        "general.finetune",
        "general.base_model.0.name",
        "general.base_model.0.repo_url",
        "general.repo_url",
    ] {
        if let Some(value) = get_gguf_string_from_map(&mmproj.kv, key) {
            combined.push_str(value);
            combined.push(' ');
        }
    }
    let normalized = normalize_alpha_num(&combined);
    let has_qwen3vl = normalized.contains("qwen3vl");
    let has_qwen35 = normalized.contains("qwen35") || normalized.contains("qwen3p5");
    (has_qwen3vl, has_qwen35)
}

pub(crate) fn validate_mmproj_for_backend(cfg: &Config, mmproj: &GGUFFile) -> Result<(), String> {
    match cfg.capabilities.multimodal_backend {
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
            if !qwen3vl::supports_qwen3vl_clip_mmproj(mmproj) {
                return Err(
                    "unsupported mmproj for this runner: expected clip.projector_type='qwen3vl_merger'"
                        .to_string(),
                );
            }
            qwen3vl::validate_mmproj_matches_text_model(cfg, mmproj)?;
            let (has_qwen3vl, has_qwen35) = mmproj_family_markers(mmproj);
            match cfg.capabilities.multimodal_backend {
                MultimodalBackend::Qwen35 => {
                    if has_qwen3vl && !has_qwen35 {
                        return Err("incompatible mmproj family for qwen35 backend: sidecar metadata indicates Qwen3-VL; use a Qwen3.5 mmproj from the same checkpoint family".to_string());
                    }
                    if mmproj
                        .tensors
                        .iter()
                        .any(|tensor| tensor.name.starts_with("v.deepstack."))
                    {
                        return Err("incompatible mmproj family for qwen35 backend: deepstack vision tensors detected (Qwen3-VL style); use a Qwen3.5 mmproj sidecar".to_string());
                    }
                }
                MultimodalBackend::Qwen3Vl => {
                    if has_qwen35 && !has_qwen3vl {
                        return Err("incompatible mmproj family for qwen3vl backend: sidecar metadata indicates Qwen3.5; use a Qwen3-VL mmproj from the same checkpoint family".to_string());
                    }
                }
                _ => {}
            }
            Ok(())
        }
        _ => Err(format!(
            "external vision mmproj is unsupported for backend '{}'",
            cfg.capabilities.multimodal_backend.as_str()
        )),
    }
}

pub(crate) fn build_vision_encoder_from_mmproj(
    cfg: &Config,
    mmproj: GGUFFile,
) -> Result<Option<VisionEncoder>, String> {
    match cfg.capabilities.multimodal_backend {
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
            validate_mmproj_for_backend(cfg, &mmproj)?;
            let encoder =
                qwen3vl::Qwen3VlVisionEncoder::new(mmproj, cfg.dim, cfg.n_deepstack_layers)?;
            Ok(Some(VisionEncoder::Qwen3Vl(encoder)))
        }
        _ => Ok(None),
    }
}
