import json
import argparse
import math
import sys
import os

def calculate_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file: {e}")
            sys.exit(1)

    if not isinstance(data, list):
        print("Error: JSON content must be a list of result objects.")
        sys.exit(1)

    if not data:
        print("No data found in the file.")
        return

    total_wer_sum = 0.0
    total_words = 0
    total_errors = 0.0
    
    count = 0
    valid_items = []

    print(f"Processing {len(data)} items from {file_path}...\n")

    for i, item in enumerate(data):
        # Check for required fields
        if 'wer' not in item:
            print(f"Warning: Item {i} (id: {item.get('id', 'unknown')}) missing 'wer' field. Skipping.")
            continue
        
        # Prefer normalized_reference, fallback to reference
        ref_text = item.get('normalized_reference', item.get('reference', ''))
        
        # Count words in reference
        # Assuming normalized_reference is already tokenized/cleaned or space-separated
        ref_words = ref_text.split()
        ref_len = len(ref_words)
        
        wer = item['wer']
        
        # Calculate errors for this item (approximate from WER * ref_len)
        # errors = S + D + I
        # WER = errors / N
        # errors = WER * N
        errors = wer * ref_len
        
        total_wer_sum += wer
        total_words += ref_len
        total_errors += errors
        count += 1
        valid_items.append(wer)

    if count == 0:
        print("No valid items found to calculate metrics.")
        return

    # Macro Average WER (Average of WERs)
    macro_wer = total_wer_sum / count

    # Micro Average WER (Total Errors / Total Words)
    # This is the "Overall WER" typically reported in corpus-level benchmarks
    if total_words > 0:
        micro_wer = total_errors / total_words
    else:
        micro_wer = 0.0

    # Standard Deviation of WER
    variance = sum((x - macro_wer) ** 2 for x in valid_items) / count
    std_dev = math.sqrt(variance)

    print("-" * 40)
    print(f"Overall Statistics ({count} samples)")
    print("-" * 40)
    print(f"Total Reference Words: {total_words}")
    print(f"Macro Average WER:     {macro_wer:.4f} ({macro_wer*100:.2f}%)")
    print(f"Micro Average WER:     {micro_wer:.4f} ({micro_wer*100:.2f}%)")
    print(f"WER Std Dev:           {std_dev:.4f}")
    print(f"Min WER:               {min(valid_items):.4f}")
    print(f"Max WER:               {max(valid_items):.4f}")
    print("-" * 40)
    
    # Interpretation
    print("\nNote:")
    print("- Macro Average WER is the average of individual WER scores.")
    print("- Micro Average WER represents the overall error rate across all words in the dataset (Total Errors / Total Words).")
    print("  This is usually the standard 'Overall' metric for ASR.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate overall WER metrics from benchmark results JSON.")
    parser.add_argument("file_path", help="Path to the benchmark results JSON file.")
    
    args = parser.parse_args()
    calculate_metrics(args.file_path)

