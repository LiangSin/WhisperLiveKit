# WhisperLiveKit Benchmark

This directory contains scripts to benchmark the WhisperLiveKit backend using the LibriSpeech dataset. It supports parallel processing to maximize throughput and efficiency.

## Files Description

- `run_benchmark.py`: The main script to run the benchmark. Connects to the WebSocket server, streams audio, and records transcripts.
- `calculate_overall_metrics.py`: A script to calculate overall Word Error Rate (WER) statistics (Micro/Macro averages) from the JSON output.
- `dataset.py`: Helper module to load and organize the LibriSpeech dataset (groups audio by chapters).
- `requirements.txt`: Python dependencies required for benchmarking.

## Prerequisites

1.  **WhisperLiveKit Backend**: You must have the backend running and accessible via WebSocket (default `ws://localhost:8000/asr`).
    ```bash
    # Example command to start the server (from root)
    # By default, the backend uses FFmpeg to process incoming audio streams.
    python3 -m whisperlivekit.basic_server --port 8000 --host 0.0.0.0
    ```

2.  **LibriSpeech Dataset**: You need the LibriSpeech dataset (e.g., `dev-clean`). The script expects the standard directory structure:
    ```
    LibriSpeech/
      dev-clean/
        SPEAKER_ID/
          CHAPTER_ID/
            SPEAKER-CHAPTER-INDEX.flac
            SPEAKER-CHAPTER.trans.txt
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
python3 run_benchmark.py --dataset_path /path/to/LibriSpeech/dev-clean --dataset_class LibriSpeechDataset
```

#### Arguments

-   `--dataset_path`: Path to the root of the LibriSpeech dataset (e.g., `/work/b11902009/LibriSpeech/LibriSpeech/dev-clean`). **Required**.
-   `--dataset_class`: Name of the dataset class in `dataset.py` (e.g. `LibriSpeechDataset`). **Required**.
-   `--url`: WebSocket URL of the WhisperLiveKit server. Default: `ws://localhost:8000/asr`.
-   `--output`: Path to save the results JSON file. Default: `benchmark_results.json`.

#### Example

```bash
python3 benchmarking/run_benchmark.py \
  --dataset_path dataset/LibriSpeech/dev-clean \
  --dataset_class LibriSpeechDataset \
  --url ws://localhost:8000/asr \
  --output benchmarking/results/dev-clean.json
```

### 2. Analyze Results

Use `calculate_overall_metrics.py` to compute detailed WER statistics (Micro/Macro averages) from the generated JSON result file.

```bash
python3 calculate_overall_metrics.py results/dev-clean.json
```

#### Output Explanation

The script provides:
- **Macro Average WER**: The average of WER scores calculated per sample.
- **Micro Average WER**: The overall error rate (Total Errors / Total Reference Words). This is usually the standard "Overall WER" metric used in academic papers.
- **Statistics**: Standard deviation, Min, and Max WER.

Example Output:
```text
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
