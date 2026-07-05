import asyncio
import websockets
import json
import os
import argparse
import sys
import ssl
import time
from tqdm import tqdm
import struct


def parse_end(value, default=0.0) -> float:
    """Convert a line's end field to float seconds.

    Line.to_dict() serialises timestamps as 'H:MM:SS' strings via format_time().
    This helper handles both that format and plain numeric values so the
    benchmark accumulation logic can compare timestamps correctly.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    # 'H:MM:SS' or 'HH:MM:SS'
    try:
        parts = str(value).split(":")
        seconds = sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
        return seconds
    except (ValueError, AttributeError):
        return default

# Ensure we can import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import DatasetClass

async def run_benchmark(dataset_path, dataset_class_name, websocket_url, output_file, debug=False, translate=False):
    print(f"Loading dataset from {dataset_path} using class {dataset_class_name}...")
    
    try:
        DSClass = getattr(DatasetClass, dataset_class_name)
    except AttributeError:
        print(f"Error: Dataset class '{dataset_class_name}' not found in DatasetClass/")
        sys.exit(1)
        
    dataset_instance = DSClass(dataset_path, translate=translate)
    print(f"Found {len(dataset_instance)} samples.")
    
    if len(dataset_instance) == 0:
        print("No samples found. Check the dataset path.")
        return
    
    results = []
    total_error_ratio = 0.0
    count = 0

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
    
    print(f"Starting benchmark against {websocket_url}...")
    
    pbar = tqdm(dataset_instance)
    
    for chapter_samples in pbar:
        if not chapter_samples:
            continue
            
        group_id = dataset_instance.get_group_id(chapter_samples)
            
        try:
            # Create a new connection for each chapter
            ssl_context = None
            if websocket_url.startswith("wss://"):
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
                # Wait for config message
                config_msg = await websocket.recv()
                try:
                    config = json.loads(config_msg)
                except json.JSONDecodeError:
                    print(f"Error decoding config message: {config_msg}")
                    config = {}

                if debug:
                    print(f"[DEBUG] Connected to {websocket_url}, config: {config}")
                    
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
                    if debug:
                        print(f"[DEBUG] Sending WAV header for group {group_id}")
                    await websocket.send(wav_header)

                # State variable to track transcript for the whole chapter
                accumulated_text = ""           # committed sentences concatenated so far
                accumulated_end = 0.0           # end timestamp of the last committed sentence
                pending_text = ""               # last (possibly in-progress) sentence from latest batch
                accumulated_translation = ""    # committed translations concatenated so far
                accumulated_translation_end = 0.0  # end timestamp of last committed translation line
                pending_translation = ""        # translation of the current pending sentence

                def update_transcript(msg_data):
                    nonlocal accumulated_text, accumulated_end, pending_text
                    nonlocal accumulated_translation, accumulated_translation_end, pending_translation
                    if "lines" not in msg_data:
                        return
                    lines = msg_data["lines"]
                    if not lines:
                        return

                    # The last line may still be a pending/in-progress sentence that
                    # will be updated in the next message; all preceding lines are stable.
                    committed_lines = lines[:-1]
                    last_line = lines[-1]

                    new_lines = [l for l in committed_lines if parse_end(l.get("end")) > accumulated_end]
                    if new_lines:
                        new_text = " ".join(l.get("text", "") for l in new_lines if l.get("text"))
                        if new_text:
                            accumulated_text = (accumulated_text + " " + new_text).strip()
                        accumulated_end = max(parse_end(l.get("end")) for l in new_lines)

                    # Track the latest pending sentence (may be replaced next message).
                    pending_text = last_line.get("text", "") if parse_end(last_line.get("end")) > accumulated_end else ""

                    if translate:
                        for l in committed_lines:
                            line_end = parse_end(l.get("end"))
                            if line_end <= accumulated_translation_end:
                                continue
                            trans = (l.get("translation") or "").strip()
                            if not trans:
                                break
                            accumulated_translation = (accumulated_translation + " " + trans).strip()
                            accumulated_translation_end = line_end

                        # Track pending translation (mirrors pending_text logic).
                        last_trans = (last_line.get("translation") or "").strip()
                        pending_translation = last_trans if parse_end(last_line.get("end")) > accumulated_translation_end else ""

                async def send_audio():
                    """Stream audio chunks to the websocket."""
                    for sample in chapter_samples:
                        process = await asyncio.create_subprocess_exec(
                            "ffmpeg", "-i", sample['audio_path'],
                            "-f", "s16le", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                            "pipe:1",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        i = 0
                        while True:
                            i += 1
                            chunk = await process.stdout.read(4096)
                            if not chunk:
                                break
                            await websocket.send(chunk)
                            if debug:
                                print(f"[DEBUG] Sent chunk {i} for group {group_id}")
                            if i % 10000 == 0:
                                await asyncio.sleep(0.05)

                    # Send End of Stream signal (Empty Bytes)
                    await websocket.send(b"")
                    if debug:
                        print(f"[DEBUG] Sent EOS for group {group_id}")

                async def receive_messages():
                    """Receive server messages until ready_to_stop."""
                    while True:
                        try:
                            msg = await websocket.recv()
                            data = json.loads(msg)
                            if debug:
                                print(f"[DEBUG] Received message for group {group_id}: {data}")
                            if data.get("type") == "ready_to_stop":
                                if debug:
                                    print(f"[DEBUG] Received ready_to_stop for group {group_id}")
                                break
                            update_transcript(data)
                        except websockets.exceptions.ConnectionClosed:
                            print(f"Connection closed while waiting to stop for {group_id}")
                            break
                        except json.JSONDecodeError as e:
                            print(f"Error decoding message for group {group_id}: {e}")
                            continue

                # Run send and receive concurrently to simulate streaming
                send_task = asyncio.create_task(send_audio())
                recv_task = asyncio.create_task(receive_messages())
                try:
                    await asyncio.gather(send_task, recv_task)
                finally:
                    # Ensure both tasks are cancelled if one fails
                    for t in (send_task, recv_task):
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(send_task, recv_task, return_exceptions=True)

                if translate and translate_out is not None:
                    full_translation = " ".join(filter(None, [accumulated_translation, pending_translation])).strip()
                    translate_out.write(full_translation + "\n")
                    translate_out.flush()

                # Write COMET reference to translate_answer_out
                if translate and translate_answer_out is not None:
                    for sample in chapter_samples:
                        trans_result = sample.get('translation')
                        if trans_result:
                            translate_answer_out.write(trans_result + "\n")
                    translate_answer_out.flush()

                # Combine committed sentences with the final pending sentence (if any).
                hyp = " ".join(filter(None, [accumulated_text, pending_text])).strip()
                # Concatenate references
                ref = " ".join([s['text'] for s in chapter_samples])
                
                norm_ref = dataset_instance.normalize(ref)
                norm_hyp = dataset_instance.normalize(hyp)

                # Delegate error metric computation to the dataset implementation
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
        finally:
            await asyncio.sleep(1)

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
        print(f"Translation answer (COMET reference) saved to {translate_answer_file}")

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
    parser.add_argument("--dataset_class", required=True, help="Name of the dataset class in DatasetClass/ (e.g. LibriSpeechDataset)")
    parser.add_argument("--url", default="ws://localhost:8000/asr", help="WebSocket URL of the running server")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--translate", action="store_true", help="Collect translation output from the server and save to a separate file")
    args = parser.parse_args()
    
    # Python 3.6 compatibility
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_benchmark(args.dataset_path, args.dataset_class, args.url, args.output, debug=args.debug, translate=args.translate))
