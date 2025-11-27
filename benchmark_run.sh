# benchmarking

DATASET=dev-clean-debug
DATASET_CLASS=LibriSpeechDataset

python3 benchmarking/run_benchmark.py \
    --dataset_path dataset/LibriSpeech/${DATASET} \
    --dataset_class ${DATASET_CLASS} \
    --url "ws://localhost:8000/asr" \
    --output "benchmarking/results/${DATASET}.json"