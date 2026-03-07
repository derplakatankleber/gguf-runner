# Image Scaling

gguf-runner automatically selects image resolution based on the loaded model and its mmproj sidecar.
The goal is to balance token budget, visual detail, and the spatial resolution the vision encoder
was trained at.

## Resolution source

When a mmproj sidecar is present the vision encoder exposes two values read directly from the
mmproj GGUF metadata:

| Key | Meaning | Typical value |
|---|---|---|
| `clip.vision.image_size` | Base training resolution (`base_size`) | 768 px |
| `clip.vision.patch_size × spatial_merge_size` | Alignment unit (`align_to`) | 28 px (14 × 2) |

All computed sizes are rounded down to the nearest multiple of `align_to`.

## Qwen3.5 — dimension-scaled resolution

Qwen3.5 multimodal models use a dynamic formula that grows the input resolution with the
language-model embedding dimension (`dim`), because larger models can usefully attend to more
visual detail without running out of context.

```
balanced_size = base_size × dim / 3072
balanced_size = clamp(balanced_size, max(align_to, 224), base_size × 2)
target        = floor(balanced_size / align_to) × align_to
```

The anchor point `3072` is the embedding dimension of a 7B-class model, which receives exactly
`base_size` pixels. Smaller models get proportionally less; larger models get proportionally more,
up to 2 × base_size.

### Concrete examples (base_size = 768, align_to = 28)

| Model size | dim  | Raw formula | Aligned target |
|---|---|---|---|
| 2B | 2048 | 768 × 2048 / 3072 = 512 | **504 px** |
| 7B (anchor) | 3072 | 768 × 3072 / 3072 = 768 | **756 px** |
| 14B | 5120 | 768 × 5120 / 3072 = 1280 | **1260 px** |
| ≥ 19B | ≥ 6144 | ≥ 1536 — hits 2 × cap | **1512 px** |

The 2 × cap (1536 px with base_size = 768) is where bilinear interpolation of the vision
encoder's learned position embeddings is still reliable.

The resize mode is **FitWithin**: the image is scaled so that the longer edge equals the target,
preserving aspect ratio, with no cropping.

## Qwen3-VL — fixed base resolution

Qwen3-VL models always use exactly `base_size` (768 px) with a **CenterCrop** resize mode.
The image is first scaled so the shorter edge equals the target, then center-cropped to a square.

## Gemma3 — fixed SigLIP resolution

Gemma3 multimodal models use the mmproj encoder `base_size` (typically 896 px) with a
**Stretch** resize mode, matching llama.cpp Gemma3 preprocessing: direct bilinear resize to
`base_size × base_size` without aspect-preserving fit or center-crop.

## Detail crop (Qwen3.5 small models only)

For Qwen3.5 models with 24 or fewer transformer layers (e.g. the 2B variant), a second image
is automatically appended to every single-image request. This secondary image is a center-square
crop of the original, giving the model a higher-detail zoomed-in view of the central region —
useful when fine print, logos, or other small details appear near the center.

The crop is skipped when:
- The model has more than 24 layers (larger Qwen3.5 variants).
- The input already contains more than one image, a video, or audio.
- The source image is already square, or its shorter side is less than 64 px.

The temporary crop file is written to the system temp directory and referenced by the prompt
with the caption `(Second image: centered close-up crop of the same source.)`.

## Fallback (no mmproj sidecar)

If no mmproj sidecar is found, a minimal 224 × 224 fallback profile is used for Qwen models.
Gemma3 fallback uses 896 × 896 with the same **Stretch** resize mode.
