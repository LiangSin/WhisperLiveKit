import argparse
import json
import os
import sys
import time
from tqdm import tqdm
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure we can import dataset and whisper modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import DatasetClass
from whisperlivekit.whisper import load_model, transcribe
from whisperlivekit.model_paths import resolve_model_path, model_path_and_type

def load_whisper_model(model_path, device="auto"):
    """
    Load Whisper model using the whisperlivekit implementation.
    Supports both built-in model names and local paths.
    """
    print(f"Loading Whisper model: {model_path}")

    # Determine device
    if device == "auto" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Handle custom model paths like whisperlivekit does
    try:
        print(f"Resolving model path: {model_path}")
        resolved_model_path = resolve_model_path(model_path)
        print(f"Resolved model path: {resolved_model_path}")

        pytorch_path, mlx_compat, fw_compat = model_path_and_type(resolved_model_path)
        print(f"PyTorch path: {pytorch_path}, MLX compatible: {mlx_compat}, FW compatible: {fw_compat}")

        if pytorch_path:
            # Use the resolved PyTorch checkpoint path (keep as Path like SimulStreamingASR does)
            model_name_or_path = pytorch_path
            download_root = resolved_model_path
            print(f"Using PyTorch checkpoint: {model_name_or_path}")
            print(f"Download root: {download_root}")
        else:
            # Fall back to original path (for built-in models)
            model_name_or_path = model_path
            download_root = None
            print(f"No PyTorch checkpoint found, using: {model_name_or_path}")

        print(f"Calling load_model with name={model_name_or_path}, device={device}, download_root={download_root}")
        model = load_model(model_name_or_path, device=device, download_root=download_root)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_audio_file(audio_path, model):
    """
    Process a single audio file and return transcription.
    """
    try:
        # Use whisperlivekit's transcribe function
        result = transcribe(model, audio_path, verbose=False, fp16=True)
        transcription = result["text"].strip()

        return transcription
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return ""

def run_nonstreaming_benchmark(dataset_path, dataset_class_name, model_path, output_file, device="auto", batch_size=1):
    """
    Run non-streaming benchmark on dataset with Whisper model.
    """
    print(f"Loading dataset from {dataset_path} using class {dataset_class_name}...")

    try:
        DSClass = getattr(DatasetClass, dataset_class_name)
    except AttributeError:
        print(f"Error: Dataset class '{dataset_class_name}' not found in DatasetClass/")
        sys.exit(1)

    dataset_instance = DSClass(dataset_path)
    print(f"Found {len(dataset_instance)} samples.")

    if len(dataset_instance) == 0:
        print("No samples found. Check the dataset path.")
        return

    # Load the model
    pipe = load_whisper_model(model_path, device)

    results = []
    total_error_ratio = 0.0
    count = 0

    print("Starting non-streaming benchmark...")

    start_time = time.time()
    pbar = tqdm(dataset_instance)

    for chapter_samples in pbar:
        if not chapter_samples:
            continue

        group_id = dataset_instance.get_group_id(chapter_samples)

        try:
            # Process all samples in the chapter
            chapter_transcripts = []

            for sample in chapter_samples:
                audio_path = sample['audio_path']

                # Get transcription for this audio file
                transcription = process_audio_file(audio_path, pipe)
                chapter_transcripts.append(transcription)

            # Combine all transcripts for the chapter
            hyp = " ".join(chapter_transcripts).strip()

            # Concatenate references
            ref = " ".join([s['text'] for s in chapter_samples])

            # Normalize texts
            norm_ref = dataset_instance.normalize(ref)
            norm_hyp = dataset_instance.normalize(hyp)

            # Compute error ratio
            error_ratio = dataset_instance.compute_error_ratio(norm_ref, norm_hyp)

            results.append({
                "id": group_id,
                "reference": ref,
                "hypothesis": hyp,
                "normalized_reference": norm_ref,
                "normalized_hypothesis": norm_hyp,
                "error_ratio": error_ratio,
                "sample_count": len(chapter_samples)
            })

            total_error_ratio += error_ratio
            count += 1
            pbar.set_description(f"Avg error_ratio: {total_error_ratio/count:.4f}")

        except Exception as e:
            print(f"Error processing group {group_id}: {e}")
            results.append({
                "id": group_id,
                "error": str(e)
            })

    end_time = time.time()
    processing_time = end_time - start_time

    if count > 0:
        avg_error_ratio = total_error_ratio / count
        print("Benchmark complete.")
        print(f"Processed {count} samples in {processing_time:.2f} seconds")
    else:
        print("\nBenchmark complete. No samples processed successfully.")

    # Save results
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-streaming benchmark for Whisper models")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root")
    parser.add_argument("--dataset_class", required=True, help="Name of the dataset class in DatasetClass/")
    parser.add_argument("--model_path", required=True, help="Path or Hugging Face model name for Whisper model")
    parser.add_argument("--output", default="nonstreaming_benchmark_results.json", help="Output file for results")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (currently not used)")

    args = parser.parse_args()

    run_nonstreaming_benchmark(
        args.dataset_path,
        args.dataset_class,
        args.model_path,
        args.output,
        args.device,
        args.batch_size
    )