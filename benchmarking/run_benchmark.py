import asyncio
import websockets
import json
import os
import argparse
import sys
import time
from tqdm import tqdm
import struct

# Ensure we can import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset

async def run_benchmark(dataset_path, dataset_class_name, websocket_url, output_file):
    print(f"Loading dataset from {dataset_path} using class {dataset_class_name}...")
    
    try:
        DatasetClass = getattr(dataset, dataset_class_name)
    except AttributeError:
        print(f"Error: Dataset class '{dataset_class_name}' not found in dataset.py")
        sys.exit(1)
        
    dataset_instance = DatasetClass(dataset_path)
    print(f"Found {len(dataset_instance)} samples.")
    
    if len(dataset_instance) == 0:
        print("No samples found. Check the dataset path.")
        return
    
    results = []
    total_error_ratio = 0.0
    count = 0
    
    print(f"Starting benchmark against {websocket_url}...")
    
    pbar = tqdm(dataset_instance)
    
    for chapter_samples in pbar:
        if not chapter_samples:
            continue
            
        group_id = dataset_instance.get_group_id(chapter_samples)
            
        try:
            # Create a new connection for each chapter
            async with websockets.connect(websocket_url) as websocket:
                # Wait for config message
                config_msg = await websocket.recv()
                try:
                    config = json.loads(config_msg)
                except json.JSONDecodeError:
                    print(f"Error decoding config message: {config_msg}")
                    config = {}
                    
                use_audio_worklet = config.get("useAudioWorklet", False)
                
                # Generate dummy WAV header for 16kHz mono s16le if needed
                if not use_audio_worklet:
                    # minimal WAV header with 0 length (some players might dislike it, but ffmpeg usually handles it)
                    # or set to max size 0xFFFFFFFF
                    data_size = 0xFFFFFFFF
                    
                    wav_header = b'RIFF' + struct.pack('<I', 36 + data_size if data_size != 0xFFFFFFFF else 0xFFFFFFFF) + b'WAVEfmt '
                    wav_header += struct.pack('<I', 16) + struct.pack('<H', 1) + struct.pack('<H', 1) 
                    wav_header += struct.pack('<I', 16000) + struct.pack('<I', 32000) + struct.pack('<H', 2) + struct.pack('<H', 16)
                    wav_header += b'data' + struct.pack('<I', data_size)
                    
                    # Send header first
                    await websocket.send(wav_header)

                # State variable to track transcript for the whole chapter
                final_transcript_for_chapter = ""

                def update_transcript(msg_data):
                    nonlocal final_transcript_for_chapter
                    if "lines" in msg_data:
                        lines = msg_data["lines"]
                        text_parts = [l.get("text", "") for l in lines if l.get("text")]
                        current_text = " ".join(text_parts)
                        if current_text:
                            final_transcript_for_chapter = current_text

                # Process all samples in the chapter
                for sample in chapter_samples:
                    # 1. Prepare audio
                    # Always convert to PCM s16le 16000Hz
                    process = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-i", sample['audio_path'], 
                        "-f", "s16le", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", 
                        "pipe:1",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    audio_data, _ = await process.communicate()
                    
                    # 2. Send audio
                    chunk_size = 4096
                    for i in range(0, len(audio_data), chunk_size):
                        await websocket.send(audio_data[i:i+chunk_size])
                        await asyncio.sleep(0.001) 
                        
                        # Consume messages while sending to update buffer
                        try:
                            while True:
                                msg = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                                data = json.loads(msg)
                                update_transcript(data)
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            pass
                
                # 3. Send End of Stream signal (Empty Bytes)
                # This triggers the server to flush buffers and finish processing
                await websocket.send(b"")

                # 4. Wait for "ready_to_stop" signal
                
                while True:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(msg)
                        
                        # Check for stop signal
                        if data.get("type") == "ready_to_stop":
                            break
                            
                        update_transcript(data)
                                
                    except asyncio.TimeoutError:
                        print(f"Timeout waiting for completion on group {group_id}")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        # Connection closed by server (could be normal or error)
                        break

                hyp = final_transcript_for_chapter.strip()
                # Concatenate references
                ref = " ".join([s['text'] for s in chapter_samples])
                
                norm_ref = dataset_instance.normalize(ref)
                norm_hyp = dataset_instance.normalize(hyp)

                # Delegate error metric computation to the dataset implementation
                error_ratio = dataset_instance.compute_error_ratio(ref, hyp)
                
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

    if count > 0:
        avg_error_ratio = total_error_ratio / count
        print(f"\nBenchmark complete. Processed {count} samples.")
        print(f"Average error_ratio: {avg_error_ratio:.4f}")
    else:
        print("\nBenchmark complete. No samples processed successfully.")

    # Save results
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark WhisperLiveKit against LibriSpeech")
    parser.add_argument("--dataset_path", required=True, help="Path to LibriSpeech dataset root (e.g. /path/to/LibriSpeech)")
    parser.add_argument("--dataset_class", required=True, help="Name of the dataset class in dataset.py (e.g. LibriSpeechDataset)")
    parser.add_argument("--url", default="ws://localhost:8000/asr", help="WebSocket URL of the running server")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Python 3.6 compatibility
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_benchmark(args.dataset_path, args.dataset_class, args.url, args.output))
