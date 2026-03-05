To download models from Hugging Face, use the `resolve/main/<filename>` pattern.

Examples:
- `wget https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- `wget https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M.gguf`

## Qwen3-VL `mmproj` (vision encoder)

Qwen3-VL GGUF is split into two artifacts:
- LLM (`Qwen3VL-...*.gguf`)
- vision encoder sidecar (`mmproj-...*.gguf`)

For multimodal image/video use, download both.

Official Qwen repos:
- `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF`
- `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF`

Example downloads (30B Instruct):
- `wget https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF/resolve/main/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- `wget https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf`

Example downloads (2B Instruct):
- `wget https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- `wget https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

Placement:
- Place the `mmproj` file in the same directory as the model file.
- `gguf-runner` auto-discovers local `mmproj*.gguf` sidecars (no extra CLI flag).

## Qwen3.5 `mmproj` (vision encoder)

Qwen3.5 GGUF multimodal use also requires a sidecar:
- LLM (`Qwen3.5-...*.gguf`)
- vision encoder sidecar (`mmproj-Qwen3.5-...*.gguf`)

Example repo:
- `https://huggingface.co/unsloth/Qwen3.5-2B-GGUF`

Example downloads:
- `wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf`
- `wget https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-Qwen3.5-2B-F16.gguf`

Notes:
- Use a same-family pair (`Qwen3.5` text model with `Qwen3.5` mmproj). Do not mix `Qwen3-VL` and `Qwen3.5` sidecars.
- Place the sidecar in the same directory as the model; discovery is automatic.
