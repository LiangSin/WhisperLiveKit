# WhisperLiveKit Benchmark

This directory contains scripts to benchmark the WhisperLiveKit backend using the LibriSpeech dataset or Youtube datasets. It supports ASR evaluation and translation evaluation.

## Files Description

- `run_benchmark.py`: Main script to benchmark the **streaming** WhisperLiveKit server. Connects via WebSocket, streams audio, and records transcripts and (optionally) translations.
- `run_nonstreaming_benchmark.py`: Benchmark script for **non-streaming** (offline) Whisper inference, used as a comparison baseline.
- `calculate_overall_metrics.py`: Computes overall error rate statistics (Macro/Micro averages) from the JSON output of `run_benchmark.py`.
- `calculate_translate_metrics.py`: Computes **chrF++** and **COMET** scores for translation evaluation, given hypothesis and reference files.
- `DatasetClass/`: Package containing dataset loader implementations.
  - `base.py`: Abstract base class defining the dataset interface.
  - `librispeech.py`: Loader for the LibriSpeech dataset.
  - `youtube.py`: Loader for YouTube-sourced audio datasets.
- `decode_unicode.py`: Utility to decode Unicode escape sequences in text files. Useful to review Chinese results.
- `requirements.txt`: Python dependencies required for benchmarking.

## Prerequisites

1. **WhisperLiveKit Backend**: The server must be running and accessible via WebSocket (default `ws://localhost:8000/asr`).
   ```bash
   python3 -m whisperlivekit.basic_server --port 8000 --host 0.0.0.0
   ```
   To enable translation output on the server side, start it with the appropriate translation arguments (e.g. `--translate`).

2. **Dataset**: 
  - LibriSpeech: The script expects the directory structure:
    ```
    LibriSpeech/
      dev-clean/
        SPEAKER_ID/
          CHAPTER_ID/
            SPEAKER-CHAPTER-INDEX.flac
            SPEAKER-CHAPTER.trans.txt
    ```
  - Youtube: The script expects the directory structure:
    ```
    Youtube/
      CHANNEL_ID/
        PLAYLIST_ID/
          VIDEO_ID.wav
          VIDEO_ID.LANG.srt
    ```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Benchmark

Use `run_benchmark.py` to perform the ASR testing.

```bash
python3 run_benchmark.py \
  --dataset_path /path/to/LibriSpeech/dev-clean \
  --dataset_class LibriSpeechDataset \
  --url ws://localhost:8000/asr \
  --output benchmarking/results/dev-clean.json
```

#### Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset_path` | Path to the dataset root | **Required** |
| `--dataset_class` | Dataset class name in `DatasetClass/` (e.g. `LibriSpeechDataset`) | **Required** |
| `--url` | WebSocket URL of the running server | `ws://localhost:8000/asr` |
| `--output` | Path to save the results JSON file | `benchmark_results.json` |
| `--debug` | Print verbose debug logs (received messages, sent chunks) | off |
| `--translate` | Collect translation output from the server and save to separate files | off |

#### Translation Output (`--translate`)

When `--translate` is passed, two additional files are created alongside the main JSON output, inside a `translate/` subdirectory:

- `translate/<stem>.txt` — model translation output (one chapter per line, sentences space-separated)
- `translate/<stem>.answer.txt` — reference translation from the dataset (one segment per line, for COMET evaluation)

For example, with `--output benchmarking/results/dev-clean.json`:
```
benchmarking/results/
  dev-clean.json                        # ASR results
  translate/
    dev-clean.txt                       # translation hypothesis
    dev-clean.answer.txt                # translation reference
```

### 2. Compute ASR Metrics

```bash
python3 calculate_overall_metrics.py results/dev-clean.json
```

#### Output

```text
----------------------------------------
Overall Statistics (3 samples)
----------------------------------------
Total Reference Words: 1568
Macro Average WER:     0.1377 (13.77%)
Micro Average WER:     0.1371 (13.71%)
WER Std Dev:           0.0387
Min WER:               0.0917
Max WER:               0.1863
----------------------------------------
```

- **Macro Average**: Mean of per-sample error rates.
- **Micro Average**: Total errors / total reference words — the standard corpus-level WER used in academic papers.

### 3. Compute Translation Metrics

```bash
python3 calculate_translate_metrics.py \
  --hypothesis benchmarking/results/translate/dev-clean.txt \
  --reference  benchmarking/results/translate/dev-clean.answer.txt
```

#### Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `--hypothesis` | `-hyp` | Path to hypothesis file (one segment per line) | **Required** |
| `--reference` | `-ref` | Path to reference file (one segment per line) | **Required** |
| `--source` | `-src` | Optional source file for COMET (one segment per line) | none |
| `--metrics` | `-m` | Metrics to compute: `chrf`, `comet`, or both | `chrf comet` |
| `--cuda` | | CUDA device id for COMET (`-1` for CPU) | `0` |

#### Output

```text
chrF++: 42.3500
COMET:  0.8712
```

- **chrF++**: Character n-gram F-score (via `sacrebleu`). Fast, no GPU needed.
- **COMET**: Neural MT evaluation metric using `Unbabel/wmt22-comet-da` (via `unbabel-comet`). Requires GPU for practical speed.

To compute only chrF++ (fast, no GPU required):
```bash
python3 calculate_translate_metrics.py \
  --hypothesis benchmarking/results/translate/dev-clean.txt \
  --reference  benchmarking/results/translate/dev-clean.answer.txt \
  --metrics chrf
```
