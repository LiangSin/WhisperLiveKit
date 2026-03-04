#!/usr/bin/env python3
"""
Calculate chrF++ and COMET scores for translation evaluation.

Expects one segment per line in hypothesis and reference files (lines must align).
For COMET, optionally provide a source file (one segment per line).
"""

import argparse
import sys


def load_lines(path):
    """Load non-empty lines from a file, stripped."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def calc_chrf(hypothesis_path, reference_path):
    """Compute chrF++ score using sacrebleu."""
    import sacrebleu

    refs = load_lines(reference_path)
    hyps = load_lines(hypothesis_path)

    if len(hyps) != len(refs):
        sys.exit(
            f"Line count mismatch: hypothesis has {len(hyps)} lines, "
            f"reference has {len(refs)} lines. Each line must align."
        )
    if not hyps:
        sys.exit("Both files must contain at least one line.")

    # sacrebleu expects refs as list of lists (one ref list per segment, or shared refs)
    # For chrF: corpus_chrf(systems, refs) where refs is [[r1],[r2],...] or [[r1,r2,...]]
    refs_as_lists = [[r] for r in refs]
    score = sacrebleu.corpus_chrf(hyps, refs_as_lists)
    return float(score.score)


def calc_comet(hypothesis_path, reference_path, source_path=None, cuda_device=0):
    """Compute COMET score using unbabel-comet."""
    import torch
    from comet import download_model, load_from_checkpoint

    hyps = load_lines(hypothesis_path)
    refs = load_lines(reference_path)

    if len(hyps) != len(refs):
        sys.exit(
            f"Line count mismatch: hypothesis has {len(hyps)} lines, "
            f"reference has {len(refs)} lines. Each line must align."
        )
    if not hyps:
        sys.exit("Both files must contain at least one line.")

    if source_path:
        srcs = load_lines(source_path)
        if len(srcs) != len(hyps):
            sys.exit(
                f"Line count mismatch: source has {len(srcs)} lines, "
                f"hypothesis has {len(hyps)} lines."
            )
    else:
        srcs = [""] * len(hyps)

    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(srcs, hyps, refs)
    ]

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    gpus = 1 if torch.cuda.is_available() and cuda_device >= 0 else 0
    devices = [cuda_device] if gpus else None
    output = model.predict(data, batch_size=8, gpus=gpus, devices=devices)
    return float(output.system_score)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate chrF++ and COMET scores for translation evaluation."
    )
    parser.add_argument(
        "--hypothesis",
        "-hyp",
        required=True,
        help="Path to hypothesis file (model output, one segment per line)",
    )
    parser.add_argument(
        "--reference",
        "-ref",
        required=True,
        help="Path to reference file (ground truth, one segment per line)",
    )
    parser.add_argument(
        "--source",
        "-src",
        default=None,
        help="Path to source file for COMET (optional, one segment per line)",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["chrf", "comet"],
        choices=["chrf", "comet"],
        help="Metrics to compute (default: chrf comet)",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device id for COMET (e.g. 1 for cuda:1, -1 for CPU). Default: 0",
    )
    args = parser.parse_args()

    results = {}

    if "chrf" in args.metrics:
        chrf_score = calc_chrf(args.hypothesis, args.reference)
        results["chrF++"] = chrf_score
        print(f"chrF++: {chrf_score:.4f}")

    if "comet" in args.metrics:
        comet_score = calc_comet(
            args.hypothesis,
            args.reference,
            source_path=args.source,
            cuda_device=args.cuda,
        )
        results["COMET"] = comet_score
        print(f"COMET: {comet_score:.4f}")

    return results


if __name__ == "__main__":
    main()
