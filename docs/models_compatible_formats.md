# Model Path Formats

The `--model-path` parameter accepts:

## File Path
- **`.pt` / `.bin` / `.safetensor` formats** Should be openable by pytorch/safetensor.

## Directory Path (recommended)
Must contain:
- **`.pt` / `.safetensors` / `pytorch_model.bin` file** (required for decoder)

May optionally contain:
- **`model.bin` / `encoder.bin` / `decoder.bin`** - a CTranslate2 (faster-whisper) model. When present, the CT2 model is used for the **encoder** while decoding stays in PyTorch — a fork optimization that speeds up inference. Disable with `--disable-fast-encoder`.
- **`weights.npz`** or **`weights.safetensors`** - MLX weights for the encoder (requires mlx-whisper)

The directory is sniffed by `whisperlivekit/model_paths.py`; the production
model (`models/cool-whisper`, a symlink into the HF cache) is such a directory.

## Hugging Face Repo ID
- Provide the repo ID (e.g. `openai/whisper-large-v3`) and WhisperLiveKit will download and cache the snapshot automatically. For gated repos, authenticate via `huggingface-cli login` first.

## Converting a HuggingFace Whisper checkpoint

HF-format Whisper checkpoints (`WhisperForConditionalGeneration`) are not
directly loadable — convert them to the OpenAI `.pt` layout with
`scripts/convert_hf_whisper.py`, and optionally add a CT2 export to the same
directory to get the fast-encoder path above.

## Alignment heads

To improve speed/reduce hallucinations, use `scripts/determine_alignment_heads.py` to determine the alignment heads for your model, and pass them with `--custom-alignment-heads`. This matters for custom models: the AlignAtt streaming policy depends on alignment heads, and without the flag they default to all heads of the last half of the decoder layers, which can be noticeably worse.
