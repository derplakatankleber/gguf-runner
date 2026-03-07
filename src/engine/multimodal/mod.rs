mod gemma3;
mod injection;
mod qwen3vl;

use crate::engine::types::{Config, GGUFFile, MultimodalBackend};
use crate::engine::vision::PreparedImageTensor;
pub(crate) use injection::{expand_prompt_with_image_embeddings, ImageEmbeddingSequence};

pub(crate) enum VisionEncoder {
    Gemma3(gemma3::Gemma3VisionEncoder),
    Qwen3Vl(qwen3vl::Qwen3VlVisionEncoder),
}

impl VisionEncoder {
    pub(crate) fn recommended_image_size(&self) -> usize {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_size(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_size(),
        }
    }

    pub(crate) fn recommended_image_alignment(&self) -> usize {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_alignment(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_alignment(),
        }
    }

    pub(crate) fn recommended_image_normalization(&self) -> ([f32; 3], [f32; 3]) {
        match self {
            VisionEncoder::Gemma3(enc) => enc.recommended_image_normalization(),
            VisionEncoder::Qwen3Vl(enc) => enc.recommended_image_normalization(),
        }
    }

    pub(crate) fn encode_images(
        &self,
        images: &[PreparedImageTensor],
    ) -> Result<Vec<ImageEmbeddingSequence>, String> {
        match self {
            VisionEncoder::Gemma3(enc) => enc.encode_images(images),
            VisionEncoder::Qwen3Vl(enc) => enc.encode_images(images),
        }
    }
}

pub(crate) fn build_vision_encoder_from_mmproj(
    cfg: &Config,
    mmproj: GGUFFile,
) -> Result<Option<VisionEncoder>, String> {
    match cfg.capabilities.multimodal_backend {
        MultimodalBackend::Gemma3 => {
            let encoder = gemma3::Gemma3VisionEncoder::new(mmproj, cfg.dim)?;
            Ok(Some(VisionEncoder::Gemma3(encoder)))
        }
        MultimodalBackend::Qwen3Vl | MultimodalBackend::Qwen35 => {
            let encoder =
                qwen3vl::Qwen3VlVisionEncoder::new(mmproj, cfg.dim, cfg.n_deepstack_layers)?;
            Ok(Some(VisionEncoder::Qwen3Vl(encoder)))
        }
        _ => Ok(None),
    }
}
