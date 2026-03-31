from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

_CUCKOO_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.dirname(_CUCKOO_DIR)
sys.path.insert(0, _CUCKOO_DIR)
sys.path.insert(0, _BENCH_ROOT)

import DatasetClass


def _instantiate_dataset(DSClass, dataset_dir: str, translate: bool):
    params = inspect.signature(DSClass.__init__).parameters
    kwargs = {}
    if "translate" in params:
        kwargs["translate"] = translate
    return DSClass(dataset_dir, **kwargs)


def resolve_dataset_class(name: str):
    try:
        return getattr(DatasetClass, name)
    except AttributeError:
        pass
    if name == "BaseDataset":
        from DatasetClass.base import BaseDataset

        return BaseDataset
    print(f"Error: Dataset class '{name}' not found in DatasetClass/")
    sys.exit(1)


def _read_crawl_transcript(crawl_dir: Path, stem: str) -> tuple[str, list[str]]:
    path = crawl_dir / f"{stem}_transcript.txt"
    if not path.is_file():
        return "no results found", []
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [s.strip() for s in lines if s.strip()]
    if not non_empty:
        return "no results found", []
    return " ".join(non_empty), lines


def _read_crawl_translation(crawl_dir: Path, stem: str) -> str:
    path = crawl_dir / f"{stem}_translation.txt"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return " ".join(s.strip() for s in lines if s.strip())


def run_cuckoo_benchmark(
    dataset_dir: str,
    crawl_results_dir: str,
    output_file: str,
    dataset_class_name: str = "YoutubeDataset",
    translate: bool = False,
):
    DSClass = resolve_dataset_class(dataset_class_name)
    dataset_instance = _instantiate_dataset(DSClass, dataset_dir, translate)
    crawl_dir = Path(crawl_results_dir)

    print(f"Loading dataset from {dataset_dir} using class {dataset_class_name}...")
    print(f"Crawl results directory: {crawl_dir.resolve()}")
    print(f"Found {len(dataset_instance)} samples.")

    if len(dataset_instance) == 0:
        print("No samples found. Check the dataset path.")
        return

    if translate:
        output_dir = os.path.dirname(output_file)
        output_stem = os.path.splitext(os.path.basename(output_file))[0]
        translate_dir = os.path.join(output_dir, "translate") if output_dir else "translate"
        os.makedirs(translate_dir, exist_ok=True)
        translate_file = os.path.join(translate_dir, output_stem + ".txt")
        translate_answer_file = os.path.join(translate_dir, output_stem + ".answer.txt")
        translate_out = open(translate_file, "w", encoding="utf-8")
        translate_answer_out = open(translate_answer_file, "w", encoding="utf-8")
    else:
        translate_out = None
        translate_answer_out = None
        translate_file = ""
        translate_answer_file = ""

    results = []
    total_error_ratio = 0.0
    count = 0

    try:
        pbar = tqdm(dataset_instance)
        for chapter_samples in pbar:
            if not chapter_samples:
                continue

            group_id = dataset_instance.get_group_id(chapter_samples)
            crawl_stem = chapter_samples[0]["id"]
            ref = " ".join(s["text"] for s in chapter_samples)

            hyp, segment_lines = _read_crawl_transcript(crawl_dir, crawl_stem)

            if translate and translate_out is not None:
                trans_text = _read_crawl_translation(crawl_dir, crawl_stem)
                translate_out.write(trans_text + "\n")
                translate_out.flush()

            if translate and translate_answer_out is not None:
                for sample in chapter_samples:
                    trans_result = sample.get("translation")
                    if trans_result:
                        translate_answer_out.write(trans_result + "\n")
                translate_answer_out.flush()

            norm_ref = dataset_instance.normalize(ref)
            norm_hyp = dataset_instance.normalize(hyp)
            error_ratio = dataset_instance.compute_error_ratio(norm_ref, norm_hyp)

            results.append(
                {
                    "id": group_id,
                    "reference": ref,
                    "hypothesis": hyp,
                    "normalized_reference": norm_ref,
                    "normalized_hypothesis": norm_hyp,
                    "error_ratio": error_ratio,
                    "sample_count": len([s for s in segment_lines if s.strip()]),
                }
            )
            total_error_ratio += error_ratio
            count += 1
            pbar.set_description(f"Avg error_ratio: {total_error_ratio / count:.4f}")

        if count > 0:
            avg_error_ratio = total_error_ratio / count
            print(f"\nBenchmark complete. Processed {count} samples.")
            print(f"Average error_ratio: {avg_error_ratio:.4f}")
        else:
            print("\nBenchmark complete. No samples processed successfully.")

        if translate_out is not None:
            translate_out.close()
            print(f"Translation results saved to {translate_file}")
        if translate_answer_out is not None:
            translate_answer_out.close()
            print(f"Translation answer (reference) saved to {translate_answer_file}")

        _save_results_json(output_file, results)
    finally:
        if translate_out is not None and not translate_out.closed:
            translate_out.close()
        if translate_answer_out is not None and not translate_answer_out.closed:
            translate_answer_out.close()


def _save_results_json(output_file: str, results: list) -> None:
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Cuckoo crawl transcripts against dataset references "
            "(expects {id}_transcript.txt / {id}_translation.txt under crawl dir)."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset root.")
    parser.add_argument("--dataset_class", default="YoutubeDataset", help="Dataset class in DatasetClass/ (e.g. YoutubeDataset, LibriSpeechDataset)")
    parser.add_argument("--output", default="benchmarking/results/cuckoo_benchmark_results.json", help="Output JSON path; translate/*.txt names use this file's basename")
    parser.add_argument("--translate", action="store_true", help="Write translation lines to translate/ next to the JSON (same layout as run_benchmark.py)")
    parser.add_argument("--crawl_results_dir", default="benchmarking/cuckoo/crawl", help="Directory with {first_sample_id}_transcript.txt and optional _translation.txt")
    args = parser.parse_args()

    run_cuckoo_benchmark(
        args.dataset_dir,
        args.crawl_results_dir,
        args.output,
        dataset_class_name=args.dataset_class,
        translate=args.translate,
    )
