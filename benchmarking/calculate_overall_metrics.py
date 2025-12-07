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

    total_error_ratio_sum = 0.0
    total_words = 0
    total_errors = 0.0
    
    count = 0
    valid_items = []

    print(f"Processing {len(data)} items from {file_path}...\n")

    for i, item in enumerate(data):
        # Check for required fields
        if 'error_ratio' not in item:
            print(f"Warning: Item {i} (id: {item.get('id', 'unknown')}) missing 'error_ratio' field. Skipping.")
            continue
        
        # Prefer normalized_reference, fallback to reference
        ref_text = item.get('normalized_reference', item.get('reference', ''))
        
        # Count words in reference
        # Assuming normalized_reference is already tokenized/cleaned or space-separated
        ref_words = ref_text.split()
        ref_len = len(ref_words)
        
        error_ratio = item['error_ratio']
        
        # Calculate errors for this item (approximate from error_ratio * ref_len)
        # errors = S + D + I
        # error_ratio = errors / N
        # errors = error_ratio * N
        errors = error_ratio * ref_len
        
        total_error_ratio_sum += error_ratio
        total_words += ref_len
        total_errors += errors
        count += 1
        valid_items.append(error_ratio)

    if count == 0:
        print("No valid items found to calculate metrics.")
        return

    # Macro Average error_ratio (Average of error_ratios)
    macro_error_ratio = total_error_ratio_sum / count

    # Micro Average error_ratio (Total Errors / Total Words)
    # This is the "Overall error_ratio" typically reported in corpus-level benchmarks
    if total_words > 0:
        micro_error_ratio = total_errors / total_words
    else:
        micro_error_ratio = 0.0

    # Standard Deviation of error_ratio
    variance = sum((x - macro_error_ratio) ** 2 for x in valid_items) / count
    std_dev = math.sqrt(variance)

    print("-" * 40)
    print(f"Overall Statistics ({count} samples)")
    print("-" * 40)
    print(f"Total Reference Words: {total_words}")
    print(f"Macro Average error_ratio:     {macro_error_ratio:.4f} ({macro_error_ratio*100:.2f}%)")
    print(f"Micro Average error_ratio:     {micro_error_ratio:.4f} ({micro_error_ratio*100:.2f}%)")
    print(f"error_ratio Std Dev:           {std_dev:.4f}")
    print(f"Min error_ratio:               {min(valid_items):.4f}")
    print(f"Max error_ratio:               {max(valid_items):.4f}")
    print("-" * 40)
    
    # Interpretation
    print("\nNote:")
    print("- Macro Average error_ratio is the average of individual error_ratio scores.")
    print("- Micro Average error_ratio represents the overall error rate across all words in the dataset (Total Errors / Total Words).")
    print("  This is usually the standard 'Overall' metric for ASR.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate overall error_ratio metrics from benchmark results JSON.")
    parser.add_argument("file_path", help="Path to the benchmark results JSON file.")
    
    args = parser.parse_args()
    calculate_metrics(args.file_path)

