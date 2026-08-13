import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from whisperlivekit.timed_objects import ASRToken, Silence, Line, FrontData, State, Transcript, ChangeSpeaker
from whisperlivekit.core import TranscriptionEngine, online_factory, online_diarization_factory, online_translation_factory
from whisperlivekit.archive_writer import ConnectionArchiveWriter
from whisperlivekit.silero_vad_iterator import FixedVADIterator, OnnxWrapper, load_jit_vad
from whisperlivekit.ten_vad_iterator import TenVADIterator
from whisperlivekit.results_formater import format_output
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.hallucination_filter import load_boh, contains_hallucination, LoopingDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker
HALLUCINATION_RESET = object()  # signals SaT/Gemma to clear state when hallucination detected
FORCE_COMMIT = object()  # lag catch-up: signals SaT to commit the pending sentence immediately

_TOKENS_TRIM_THRESHOLD = 1000
_TOKENS_KEEP = 750

def cut_at(cumulative_pcm, cut_sec):
    cumulative_len = 0
    cut_sample = int(cut_sec * 16000)
    
    for ind, pcm_array in enumerate(cumulative_pcm):
        if (cumulative_len + len(pcm_array)) >= cut_sample:
            cut_chunk = cut_sample - cumulative_len
            before = np.concatenate(cumulative_pcm[:ind] + [cumulative_pcm[ind][:cut_chunk]])
            after = [cumulative_pcm[ind][cut_chunk:]] + cumulative_pcm[ind+1:]
            return before, after
        cumulative_len += len(pcm_array)
    return np.concatenate(cumulative_pcm), []

MIN_DURATION_REAL_SILENCE = 5

async def drain_queue_nowait(queue):
    items = []
    try:
        while True:
            item = queue.get_nowait()
            items.append(item)
    except asyncio.QueueEmpty:
        pass
    return items

# Cap on how much audio a single batch may concatenate. SimulStreaming's
# AlignAtt policy is designed for incremental ~0.5s inserts; when the
# producer outpaces decoding (buffered input, speed>1 replay), an uncapped
# batch grows to minutes of audio per infer and decoding quality collapses.
MAX_BATCH_SAMPLES = 8000  # 0.5 s at 16 kHz

async def get_all_from_queue(queue):
    """Await one item, then batch immediately-available ndarray items
    up to MAX_BATCH_SAMPLES of audio.

    Silence events and SENTINEL are returned alone so boundaries keep their
    position in the stream; contiguous audio chunks are concatenated.
    """
    items = []

    first_item = await queue.get()
    queue.task_done()
    if first_item is SENTINEL:
        return first_item
    if isinstance(first_item, Silence):
        return first_item
    items.append(first_item)
    batched_samples = len(first_item) if isinstance(first_item, np.ndarray) else 0

    while True:
        if not queue._queue:
            break
        next_item = queue._queue[0]
        if next_item is SENTINEL:
            break
        if isinstance(next_item, Silence):
            break
        if isinstance(next_item, np.ndarray) and batched_samples + len(next_item) > MAX_BATCH_SAMPLES:
            break
        items.append(await queue.get())
        queue.task_done()
        if isinstance(next_item, np.ndarray):
            batched_samples += len(next_item)
    if isinstance(items[0], np.ndarray):
        return np.concatenate(items)
    else: #translation
        return items

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs):
        """Initialize the audio processor with configuration, models, and state."""
        
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)
        
        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        chunk_seconds = self.args.vac_chunk_size if self.args.vac else self.args.min_chunk_size
        self.samples_per_sec = int(self.sample_rate * chunk_seconds)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = self.args.pcm_input

        # State management
        self.is_stopping = False
        self.current_silence = None
        self.total_pcm_samples = 0
        # Tail of audio dropped during silence, kept so a speech start that
        # back-dates into an already-dropped chunk can recover its onset.
        self._silence_preroll = np.array([], dtype=np.float32)
        self._preroll_max_samples = int(0.5 * self.sample_rate)
        # Audio withheld while the VAD is waiting out min_silence: it is
        # trailing silence unless speech resumes. Sending it to the ASR and
        # then force-finalizing at the boundary makes Whisper decode over
        # silence, which induces looping hallucinations.
        self._holdback = np.array([], dtype=np.float32)
        self.state = State()
        self.lock = asyncio.Lock()
        self.sep = " "  # Default separator
        self.last_response_content = FrontData()
        self.last_detected_speaker = None
        self.speaker_languages = {}
        self.diarization_before_transcription = False

        self.segments = []
        

        if self.diarization_before_transcription:
            self.cumulative_pcm = []
            self.last_start = 0.0
            self.last_end = 0.0
        
        # Hallucination filtering
        self.boh_phrases = load_boh()
        # Rolling text window: catches BoH phrases split across chunk boundaries.
        # Window size = 3x the longest phrase (min 200 chars).
        max_phrase_len = max((len(p) for p in self.boh_phrases), default=0)
        self._boh_window_size = max(max_phrase_len * 3, 200)
        self._boh_recent_text = ""
        # Stateful rolling-hash detector for triple-repetition patterns.
        # Fed only with new_text each call (O(|new_text|) vs O(window)).
        self._repeat_detector = LoopingDetector()
        # Last point we salvaged audio from after a BoH/loop reset. If the
        # decoder loops again without having advanced past it, the audio
        # itself is the trigger — skip it instead of re-decoding forever.
        self._boh_last_resume = float("-inf")
        logger.info(
            "[BoH] Hallucination filter: %s.",
            f"active with {len(self.boh_phrases)} phrase(s), window={self._boh_window_size} chars"
            if self.boh_phrases else "disabled",
        )

        # Lag catch-up (reliability): stream-time seconds the producers have
        # put into transcription_queue. The consumer compares this against how
        # much it has ingested to measure the true unprocessed backlog —
        # wall-clock lag alone cannot distinguish "inference too slow" from
        # "client stopped sending".
        self.max_transcription_lag = getattr(self.args, "max_transcription_lag", 10.0)
        self._enqueued_stream_time = 0.0
        # After a catch-up jump, end_silence()'s token-anchored time offset
        # must never rewind timestamps to before the jump.
        self._catchup_time_floor = 0.0

        # Models and processing
        self.asr = models.asr
        self.vac = None
        if self.args.vac:
            vac_backend = getattr(self.args, 'vac_backend', 'silero')
            if vac_backend == 'ten-vad':
                logger.info("Use ten-vad")
                # TEN VAD reacts at 16 ms hops, far faster than Silero's 512-sample
                # windows, so with the Silero-tuned defaults every inter-phrase pause
                # becomes a silence boundary. Each boundary force-finalizes the ASR
                # (start_silence -> process_iter(is_last=True)), which damages accuracy
                # around the cut, so boundaries must only appear at real silences:
                # require 1 s of silence to close a segment, close only when the
                # probability drops well below the trigger level (wide hysteresis,
                # so quiet speech can't end a segment it couldn't have started),
                # and back-date speech starts by 200 ms so detection latency
                # doesn't clip utterance onsets.
                self.vac = TenVADIterator(
                    threshold=self.args.vad_threshold,
                    min_silence_duration_ms=1000,
                    speech_pad_ms=200,
                    hysteresis=0.25,
                )
            else:
                # Same rationale as TEN VAD above: fewer boundaries mean fewer
                # forced finalizations. Measured on real session audio, Silero
                # with these settings emits zero non-speech audio (music /
                # ambient noise stays at prob ~0.0) and only ~3 boundaries.
                if models.vac_session is not None:
                    vac_model = OnnxWrapper(session=models.vac_session)
                else:
                    vac_model = load_jit_vad()
                self.vac = FixedVADIterator(
                    vac_model,
                    threshold=self.args.vad_threshold,
                    min_silence_duration_ms=1000,
                    speech_pad_ms=200,
                )
                         
        self.ffmpeg_manager = None
        self.ffmpeg_reader_task = None
        self._ffmpeg_error = None

        if not self.is_pcm_input:
            self.ffmpeg_manager = FFmpegManager(
                sample_rate=self.sample_rate,
                channels=self.channels
            )
            async def handle_ffmpeg_error(error_type: str):
                logger.error(f"FFmpeg error: {error_type}")
                self._ffmpeg_error = error_type
            self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
             
        # Determine which optional pipeline stages are needed
        self.use_offline_translation = (
            bool(self.args.target_language)
            and getattr(self.args, "translation_model", "nllw") != "nllw"
        )
        use_sentence_detection = getattr(self.args, "sentence_detection", False) or self.use_offline_translation

        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.translation_queue = asyncio.Queue() if self.args.target_language else None
        self.sat_queue = asyncio.Queue() if use_sentence_detection else None
        self.pcm_buffer = bytearray()

        self.transcription_task = None
        self.diarization_task = None
        self.translation_task = None
        self.sentence_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        
        self.transcription = None
        self.translation = None
        self.diarization = None
        self.sentence_detector = None
        self.archive_writer = None

        if getattr(self.args, "archive_enabled", False):
            session_dir_name = kwargs.get("session_dir_name")
            if session_dir_name:
                self.archive_writer = ConnectionArchiveWriter(
                    archive_dir=self.args.archive_dir,
                    session_dir_name=session_dir_name,
                    segment_seconds=self.args.archive_segment_seconds,
                    subtitle_flush_seconds=self.args.archive_subtitle_flush_seconds,
                    audio_format=self.args.archive_audio_format,
                )

        if self.args.transcription:
            self.transcription = online_factory(
                self.args,
                models.asr,
                init_prompt=kwargs.get("init_prompt"),
                static_init_prompt=kwargs.get("static_init_prompt"),
            )
            self.sep = self.transcription.asr.sep
        if self.args.diarization:
            self.diarization = online_diarization_factory(self.args, models.diarization_model)
        if models.translation_model:
            if self.args.translation_model == "translategemma":
                from whisperlivekit.translategemma import GemmaTranslationProcessor
                self.translation = GemmaTranslationProcessor(
                    client=models.translation_model,
                    translation_sentence_queue=self.translation_queue,
                    state=self.state,
                    lock=self.lock,
                    sentinel=SENTINEL,
                    context_sentences=getattr(self.args, "translation_context_sentences", 2),
                )
            else:
                self.translation = online_translation_factory(self.args, models.translation_model)

        if use_sentence_detection and models.sat_model is not None:
            from whisperlivekit.sentence_detector import (
                StreamingSentenceDetector,
                SentenceDetectionProcessor,
            )
            self.sentence_detector = StreamingSentenceDetector(
                sat_model=models.sat_model,
                max_tokens=getattr(self.args, "sat_max_tokens", 60),
                soft_max_tokens=getattr(self.args, "sat_soft_max_tokens", 40),
            )
            if self.use_offline_translation:
                self._sentence_proc = SentenceDetectionProcessor(
                    detector=self.sentence_detector,
                    sat_queue=self.sat_queue,
                    state=self.state,
                    lock=self.lock,
                    sentinel=SENTINEL,
                    translation_sentence_queue=self.translation_queue,
                    hallucination_reset=HALLUCINATION_RESET,
                    force_commit=FORCE_COMMIT,
                    translate_pending=(
                        getattr(self.args, "translate_pending", False)
                        and getattr(self.args, "send_pending", True)
                    ),
                    pending_translation_interval=getattr(
                        self.args, "pending_translation_interval", 1.5
                    ),
                    silence_commit_timeout=getattr(
                        self.args, "silence_commit_timeout", 2.0
                    ),
                )
            else:
                self._sentence_proc = SentenceDetectionProcessor(
                    detector=self.sentence_detector,
                    sat_queue=self.sat_queue,
                    state=self.state,
                    lock=self.lock,
                    sentinel=SENTINEL,
                    hallucination_reset=HALLUCINATION_RESET,
                    force_commit=FORCE_COMMIT,
                    silence_commit_timeout=getattr(
                        self.args, "silence_commit_timeout", 2.0
                    ),
                )
        else:
            self._sentence_proc = None

    async def _push_silence_event(self, silence_buffer: Silence):
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(silence_buffer)
            if silence_buffer.has_ended and silence_buffer.duration:
                self._enqueued_stream_time += silence_buffer.duration
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(silence_buffer)
        if self.translation_queue and not self.use_offline_translation:
            await self.translation_queue.put(silence_buffer)
        if self.sat_queue:
            await self.sat_queue.put(silence_buffer)

    async def _begin_silence(self, at_sample=None):
        if self.current_silence:
            return
        # Use audio stream time (sample-precise) for accurate silence duration
        if at_sample is not None:
            audio_t = at_sample / self.sample_rate
        else:
            audio_t = self.total_pcm_samples / self.sample_rate if self.sample_rate else 0.0
        self.current_silence = Silence(is_starting=True, start=audio_t)
        # Push a separate start-only event so _end_silence won't mutate it
        start_event = Silence(is_starting=True, start=audio_t)
        await self._push_silence_event(start_event)

    async def _end_silence(self, at_sample=None):
        if not self.current_silence:
            return
        if at_sample is not None:
            audio_t = at_sample / self.sample_rate
        else:
            audio_t = self.total_pcm_samples / self.sample_rate if self.sample_rate else 0.0
        # Guard monotonicity: clamping and pre-roll back-dating must never
        # produce a negative silence duration.
        if self.current_silence.start is not None:
            audio_t = max(audio_t, self.current_silence.start)
        self.current_silence.end = audio_t
        self.current_silence.is_starting = False
        self.current_silence.has_ended = True
        self.current_silence.compute_duration()
        if self.current_silence.duration and self.current_silence.duration > MIN_DURATION_REAL_SILENCE:
            self.state.new_tokens.append(self.current_silence)
        # Push the completed silence as the end event (separate from the start event)
        await self._push_silence_event(self.current_silence)
        self.current_silence = None

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.state.beg_loop
            self.state.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    def _trim_state_tokens_locked(self):
        """Remove old tokens to prevent unbounded growth. Caller must hold self.lock."""
        n = len(self.state.tokens)
        if n <= _TOKENS_TRIM_THRESHOLD:
            return
        excess = n - _TOKENS_KEEP
        self.state.tokens = self.state.tokens[excess:]
        self.state.last_validated_token = max(0, self.state.last_validated_token - excess)
        if self.state.last_punctuation_index is not None:
            self.state.last_punctuation_index = max(0, self.state.last_punctuation_index - excess)
        logger.debug("Trimmed %d old tokens, keeping %d", excess, _TOKENS_KEEP)

    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            remaining_transcription = 0
            if self.state.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.state.beg_loop - self.state.end_buffer, 1))
                
            remaining_diarization = 0
            if self.state.tokens:
                latest_end = max(self.state.end_buffer, self.state.tokens[-1].end if self.state.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.state.end_attributed_speaker, 1))
                
            self.state.remaining_time_transcription = remaining_transcription
            self.state.remaining_time_diarization = remaining_diarization
            
            return self.state

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it into the PCM pipeline."""
        beg = time()
        while True:
            try:
                state = await self.ffmpeg_manager.get_state() if self.ffmpeg_manager else FFmpegState.STOPPED
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    if self.is_stopping:
                        break
                    await asyncio.sleep(0.1)
                    continue

                current_time = time()
                elapsed_time = max(0.0, current_time - beg)
                buffer_size = max(int(32000 * elapsed_time), 4096)  # dynamic read
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                if chunk is None:
                    if self.is_stopping:
                        break
                    await asyncio.sleep(0.05)
                    continue
                if not chunk:
                    # Empty bytes = EOF on FFmpeg stdout; all data has been read.
                    logger.info("FFmpeg stdout EOF reached.")
                    break

                self.pcm_buffer.extend(chunk)
                await self.handle_pcm_data()

            except asyncio.CancelledError:
                logger.info("ffmpeg_stdout_reader cancelled.")
                break
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.2)

        # Flush any sub-threshold PCM data remaining in the buffer.
        await self._flush_remaining_pcm()

        logger.info("FFmpeg stdout processing finished. Signaling downstream processors if needed.")
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)

    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        cumulative_pcm_duration_stream_time = 0.0
        # Accumulate at least min_chunk_size of audio before decoding.
        # Control events (Silence/SENTINEL/...) flush the accumulator
        # first so boundaries keep their position.
        min_process_samples = int(self.sample_rate * self.args.min_chunk_size)
        pending_audio = []
        pending_samples = 0
        carry_item = None

        while True:
            try:
                if carry_item is not None:
                    item = carry_item
                    carry_item = None
                else:
                    # Use a timeout so we periodically wake up and refresh the
                    # buffer state even when no audio is flowing (e.g. silence).
                    try:
                        item = await asyncio.wait_for(
                            get_all_from_queue(self.transcription_queue),
                            timeout=0.5,
                        )
                    except asyncio.TimeoutError:
                        if pending_audio:
                            # Audio stopped mid-accumulation (e.g. speech just
                            # ended): decode the partial quantum now.
                            item = np.concatenate(pending_audio)
                            pending_audio = []
                            pending_samples = 0
                        else:
                            # No new audio — just refresh buffer state
                            _buffer_transcript = self.transcription.get_buffer()
                            async with self.lock:
                                self.state.buffer_transcription = _buffer_transcript
                            continue

                if isinstance(item, np.ndarray):
                    pending_audio.append(item)
                    pending_samples += len(item)
                    if pending_samples < min_process_samples:
                        continue
                    item = np.concatenate(pending_audio)
                    pending_audio = []
                    pending_samples = 0
                elif pending_audio:
                    # A control event arrived while audio is buffered: decode
                    # the buffered audio first, then handle the event.
                    carry_item = item
                    item = np.concatenate(pending_audio)
                    pending_audio = []
                    pending_samples = 0

                if item is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    # Flush remaining uncommitted tokens so the tail of the transcription is not silently lost.
                    if hasattr(self.transcription, 'finish'):
                        remaining_tokens, final_processed_upto = await asyncio.to_thread(
                            self.transcription.finish
                        )
                    else:
                        remaining_tokens, final_processed_upto = await asyncio.to_thread(
                            self.transcription.process_iter, True
                        )
                    if remaining_tokens:
                        remaining_tokens = list(remaining_tokens)
                        async with self.lock:
                            self.state.tokens.extend(remaining_tokens)
                            self.state.buffer_transcription = Transcript()
                            self.state.end_buffer = max(self.state.end_buffer, final_processed_upto)
                        if self.translation_queue and not self.use_offline_translation:
                            for token in remaining_tokens:
                                await self.translation_queue.put(token)
                        if self.sat_queue:
                            await self.sat_queue.put(list(remaining_tokens))
                    break

                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.state.beg_loop - self.state.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"

                # --- Lag catch-up (reliability) --------------------------------
                # Trigger only when BOTH hold:
                #  - wall-clock lag: we are behind the live edge (false during
                #    faster-than-realtime replay, e.g. benchmarks);
                #  - real backlog: audio the producers enqueued but we have not
                #    ingested yet (false when the lag comes from a client that
                #    simply stopped sending — nothing to skip there, and acting
                #    on it would reset the decoder in a loop).
                # Skipping the backlog drives the second condition back to ~0,
                # so a single catch-up cannot retrigger itself.
                backlog_s = max(0.0, self._enqueued_stream_time - cumulative_pcm_duration_stream_time)
                if (
                    self.max_transcription_lag > 0
                    and not self.diarization_before_transcription
                    and hasattr(self.transcription, "force_refresh")
                    and transcription_lag_s > self.max_transcription_lag
                    and backlog_s > self.max_transcription_lag
                ):
                    # 1) Final decode pass over the audio already inside the
                    #    decoder: commits the current hypothesis without the
                    #    AlignAtt holdback, so nothing already heard is lost.
                    forced_tokens, _ = await asyncio.to_thread(self.transcription.process_iter, True)
                    forced_tokens = list(forced_tokens or [])

                    # 2) Skip the backlog: drain the queue, keeping only the
                    #    time/speaker bookkeeping of what is thrown away.
                    skipped_s = 0.0
                    last_speaker = None
                    sentinel_seen = False

                    def _account(obj):
                        nonlocal skipped_s, last_speaker, sentinel_seen
                        if isinstance(obj, np.ndarray):
                            skipped_s += len(obj) / self.sample_rate
                        elif isinstance(obj, Silence):
                            if obj.has_ended and obj.duration:
                                skipped_s += obj.duration
                        elif isinstance(obj, ChangeSpeaker):
                            last_speaker = obj
                        elif obj is SENTINEL:
                            sentinel_seen = True

                    _account(item)
                    if carry_item is not None:
                        _account(carry_item)
                        carry_item = None
                    while not sentinel_seen:
                        try:
                            nxt = self.transcription_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        self.transcription_queue.task_done()
                        _account(nxt)

                    # 3) Jump the stream-time bookkeeping over the skipped span
                    #    and reset the decoder there: subsequent tokens are
                    #    stamped from the live edge, monotonically after the
                    #    committed ones. The skipped span remains a gap in the
                    #    transcript — by design.
                    cumulative_pcm_duration_stream_time += skipped_s
                    new_offset = cumulative_pcm_duration_stream_time
                    self.transcription.force_refresh(current_time_offset=new_offset)
                    self.transcription.end = new_offset
                    self._catchup_time_floor = new_offset
                    if last_speaker is not None:
                        self.transcription.model.speaker = last_speaker.speaker
                    # The rolling hallucination window spans the jump; clear it
                    # like the BoH reset path does.
                    self._boh_recent_text = ""
                    self._repeat_detector.reset()

                    async with self.lock:
                        self.state.tokens.extend(forced_tokens)
                        self.state.buffer_transcription = Transcript()
                        self.state.end_buffer = new_offset
                        self._trim_state_tokens_locked()

                    if self.translation_queue and not self.use_offline_translation:
                        for token in forced_tokens:
                            await self.translation_queue.put(token)
                    if self.sat_queue:
                        if forced_tokens:
                            await self.sat_queue.put(list(forced_tokens))
                        await self.sat_queue.put(FORCE_COMMIT)

                    logger.warning(
                        "[lag catch-up] lag=%.1fs backlog=%.1fs > %.1fs: committed %d token(s), "
                        "skipped %.1fs of audio, resuming at t=%.1fs.",
                        transcription_lag_s, backlog_s, self.max_transcription_lag,
                        len(forced_tokens), skipped_s, new_offset,
                    )
                    if sentinel_seen:
                        await self.transcription_queue.put(SENTINEL)
                    continue

                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                new_tokens = []
                current_audio_processed_upto = self.state.end_buffer

                if isinstance(item, Silence):
                    if item.is_starting:
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                            self.transcription.start_silence
                        )
                        asr_processing_logs += f" + Silence starting"
                    if item.has_ended:
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                        cumulative_pcm_duration_stream_time += item.duration
                        current_audio_processed_upto = cumulative_pcm_duration_stream_time
                        # Anchor on the last token end, but never before the
                        # catch-up floor: right after a lag catch-up the last
                        # token still carries a pre-jump timestamp, and using
                        # it alone would rewind global_time_offset.
                        self.transcription.end_silence(item.duration, max(
                            self.state.tokens[-1].end if self.state.tokens else 0,
                            self._catchup_time_floor,
                        ))
                    if self.state.tokens:
                        asr_processing_logs += f" | last_end = {self.state.tokens[-1].end} |"
                    logger.info(asr_processing_logs)
                    new_tokens = new_tokens or []
                    current_audio_processed_upto = max(current_audio_processed_upto, stream_time_end_of_current_pcm)
                elif isinstance(item, ChangeSpeaker):
                    self.transcription.new_speaker(item)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                    logger.info(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += len(pcm_array) / self.sample_rate
                    stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                    self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                    new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)
                    new_tokens = new_tokens or []
                else:
                    continue

                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = _buffer_transcript.text

                # Hallucination detection
                # 1. recent-tokens window: Feed only new_text into the 
                #    rolling-hash detector. Also catches BoH phrases 
                #    split across consecutive process_iter() chunks.
                # 2. buffer text: BoH phrase match only.
                if self.boh_phrases:
                    new_text = self.sep.join(t.text for t in new_tokens)
                    self._boh_recent_text = (
                        self._boh_recent_text + new_text
                    )[-self._boh_window_size:]

                    for check_text, label, det, nt in (
                        (self._boh_recent_text, "recent tokens", self._repeat_detector, new_text),
                        (buffer_text,           "buffer",        None,                  ""),
                    ):
                        is_hallucination, matched_phrase = contains_hallucination(
                            check_text, self.boh_phrases,
                            detector=det, new_text=nt,
                        )
                        if is_hallucination:
                            logger.warning(
                                "[BoH] HALLUCINATION DETECTED | source=%s | phrase=%r | text=%r",
                                label,
                                matched_phrase,
                                check_text[:120],
                            )
                            resume_from = next(
                                (t.end for t in reversed(self.state.tokens) if t.end is not None),
                                0.0,
                            )
                            if resume_from <= self._boh_last_resume + 1.0:
                                # Second trigger without progress: the audio
                                # itself induces the loop — skip it (old behavior).
                                self.transcription.force_refresh(current_time_offset=self.state.end_buffer)
                                logger.warning(
                                    "[BoH] repeat trigger without progress at t=%.2fs — skipping audio (stop-loss).",
                                    resume_from,
                                )
                            else:
                                salvaged = self.transcription.force_refresh(
                                    current_time_offset=self.state.end_buffer,
                                    resume_from=resume_from,
                                )
                                self._boh_last_resume = resume_from
                                logger.warning(
                                    "[BoH] re-queued %.2fs of audio from t=%.2fs for clean re-decode.",
                                    salvaged,
                                    resume_from,
                                )
                            self._boh_recent_text = ""
                            self._repeat_detector.reset()
                            logger.warning(
                                "[BoH] Context force-refreshed. Discarding %d token(s): %s",
                                len(new_tokens),
                                new_text,
                            )
                            new_tokens = []
                            _buffer_transcript = self.transcription.get_buffer()
                            buffer_text = _buffer_transcript.text
                            current_audio_processed_upto = self.state.end_buffer
                            if self.sat_queue:
                                await self.sat_queue.put(HALLUCINATION_RESET)
                            break

                # Rewind-storm recovery: the decoder flags deterministic
                # re-failure on the same audio (repeated same-position rewinds).
                # Same medicine as a BoH hit — clean context + salvage the
                # un-delivered audio — with the same stop-loss.
                _decoder_state = getattr(getattr(self.transcription, "model", None), "state", None)
                if _decoder_state is not None and getattr(_decoder_state, "rewind_storm", False):
                    _decoder_state.rewind_storm = False
                    resume_from = next(
                        (t.end for t in reversed(new_tokens) if t.end is not None), None)
                    if resume_from is None:
                        resume_from = next(
                            (t.end for t in reversed(self.state.tokens) if t.end is not None), 0.0)
                    if resume_from <= self._boh_last_resume + 1.0:
                        self.transcription.force_refresh(current_time_offset=self.state.end_buffer)
                        logger.warning(
                            "[rewind storm] repeat trigger without progress at t=%.2fs — skipping audio (stop-loss).",
                            resume_from,
                        )
                    else:
                        salvaged = self.transcription.force_refresh(
                            current_time_offset=self.state.end_buffer,
                            resume_from=resume_from,
                        )
                        self._boh_last_resume = resume_from
                        logger.warning(
                            "[rewind storm] context refreshed; re-queued %.2fs of audio from t=%.2fs.",
                            salvaged,
                            resume_from,
                        )
                    current_audio_processed_upto = self.state.end_buffer

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.state.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                async with self.lock:
                    self.state.tokens.extend(new_tokens)
                    self.state.buffer_transcription = _buffer_transcript
                    self.state.end_buffer = max(candidate_end_times)
                    self._trim_state_tokens_locked()

                if self.translation_queue and not self.use_offline_translation:
                    for token in new_tokens:
                        await self.translation_queue.put(token)

                if self.sat_queue and new_tokens:
                    await self.sat_queue.put(list(new_tokens))

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        if self.is_stopping:
            logger.info("Transcription processor finishing due to stopping flag.")
        if self.diarization_queue:
            await self.diarization_queue.put(SENTINEL)
        if self.translation_queue and not self.use_offline_translation:
            await self.translation_queue.put(SENTINEL)
        if self.sat_queue:
            await self.sat_queue.put(SENTINEL)

        logger.info("Transcription processor task finished.")


    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        if self.diarization_before_transcription:
            self.current_speaker = 0
            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=0.0))
        while True:
            try:
                item = await get_all_from_queue(self.diarization_queue)
                if item is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    break
                elif isinstance(item, Silence):
                    if item.has_ended:
                        diarization_obj.insert_silence(item.duration)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                else:
                    raise Exception('item should be pcm_array') 
                
                
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                if self.diarization_before_transcription:
                    segments = diarization_obj.get_segments()
                    self.cumulative_pcm.append(pcm_array)
                    if segments:
                        last_segment = segments[-1]                    
                        if last_segment.speaker != self.current_speaker:
                            cut_sec = last_segment.start - self.last_end
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            
                            self.current_speaker = last_segment.speaker
                            await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=last_segment.start))
                            
                            cut_sec = last_segment.end - last_segment.start
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)                            
                            self.last_start = last_segment.start
                            self.last_end = last_segment.end
                        else:
                            cut_sec = last_segment.end - self.last_end
                            to_transcript, self.cumulative_pcm = cut_at(self.cumulative_pcm, cut_sec)
                            await self.transcription_queue.put(to_transcript)
                            self.last_end = last_segment.end
                elif not self.diarization_before_transcription:           
                    async with self.lock:
                        self.state.tokens = diarization_obj.assign_speakers_to_tokens(
                            self.state.tokens,
                            use_punctuation_split=self.args.punctuation_split
                        )
                if len(self.state.tokens) > 0:
                    self.state.end_attributed_speaker = max(self.state.tokens[-1].end, self.state.end_attributed_speaker)

            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.info("Diarization processor task finished.")

    async def translation_processor(self):
        # the idea is to ignore diarization for the moment. We use only transcription tokens. 
        # And the speaker is attributed given the segments used for the translation
        # in the future we want to have different languages for each speaker etc, so it will be more complex.
        try:
            # Optional dependency; only used for silence-driven flush heuristics
            from nllw import MIN_SILENCE_DURATION_DEL_BUFFER
        except Exception:
            MIN_SILENCE_DURATION_DEL_BUFFER = 1.0

        # NLLW internally gates translation with: `len(buffer_text.strip().split()) >= 3`.
        # For CJK languages, ASR tokens often come without spaces, so `.split()` returns 1 and
        # translation never triggers. To make translation usable, we feed NLLW TimedText tokens
        # with simple whitespace separation between non-punctuation tokens.
        try:
            from nllw.timed_text import TimedText as NLLWTimedText
        except Exception:
            NLLWTimedText = None

        async def _flush_translation_buffer():
            """Flush any remaining tokens from the NLLW buffer into validated segments."""
            try:
                flushed, empty = self.translation.validate_buffer_and_reset()
                if flushed and getattr(flushed, "text", ""):
                    logger.debug("Translation processor: flushed remaining buffer on sentinel: %r", flushed.text[:80])
                    async with self.lock:
                        existing = self.state.translation_validated_segments
                        if not isinstance(existing, list):
                            existing = [existing] if existing else []
                        existing.append(flushed)
                        self.state.translation_validated_segments = existing[-200:]
                        self.state.buffer_translation = empty
            except Exception as e:
                logger.warning("Exception while flushing translation buffer on sentinel: %s", e)

        while True:
            try:
                item = await self.translation_queue.get() #block until at least 1 token
                if item is SENTINEL:
                    logger.debug("Translation processor received sentinel. Finishing.")
                    self.translation_queue.task_done()
                    await _flush_translation_buffer()
                    break
                elif isinstance(item, Silence):
                    # Never forward Silence objects into NLLW buffers; its backend expects tokens with `.text`.
                    # We only use ended silences as a heuristic boundary to flush/reset translation state.
                    if getattr(item, "has_ended", False) and item.duration and item.duration >= MIN_SILENCE_DURATION_DEL_BUFFER:
                        flushed, empty = self.translation.validate_buffer_and_reset(item.duration)
                        if flushed and getattr(flushed, "text", ""):
                            async with self.lock:
                                existing = self.state.translation_validated_segments
                                if not isinstance(existing, list):
                                    existing = [existing] if existing else []
                                existing.append(flushed)
                                # Keep memory bounded
                                self.state.translation_validated_segments = existing[-200:]
                                self.state.buffer_translation = empty
                    self.translation_queue.task_done()
                    continue
                
                # get all the available tokens for translation. The more words, the more precise
                tokens_to_process = [item]
                additional_tokens = await drain_queue_nowait(self.translation_queue)
                
                sentinel_found = False
                flush_on_silence = False
                for additional_token in additional_tokens:
                    if additional_token is SENTINEL:
                        sentinel_found = True
                        break
                    elif isinstance(additional_token, Silence):
                        # Filter out Silence tokens entirely (including "silence starting"), and use
                        # ended silences as a signal to flush/reset after processing the batch.
                        if getattr(additional_token, "has_ended", False) and additional_token.duration and additional_token.duration >= MIN_SILENCE_DURATION_DEL_BUFFER:
                            flush_on_silence = True
                        continue
                    else:
                        tokens_to_process.append(additional_token)                
                if tokens_to_process:
                    if NLLWTimedText is not None:
                        punctuation_marks = {".", "!", "?", "。", "！", "？"}
                        normalized_tokens = []
                        for t in tokens_to_process:
                            txt = getattr(t, "text", "") or ""
                            if not txt:
                                continue
                            try:
                                is_punct = bool(getattr(t, "is_punctuation", lambda: False)())
                            except Exception:
                                is_punct = txt.strip() in punctuation_marks
                            if not is_punct and not txt.endswith(" "):
                                txt = txt + " "
                            normalized_tokens.append(
                                NLLWTimedText(
                                    text=txt,
                                    start=getattr(t, "start", 0) or 0,
                                    end=getattr(t, "end", 0) or 0,
                                )
                            )
                        self.translation.insert_tokens(normalized_tokens)
                    else:
                        self.translation.insert_tokens(tokens_to_process)
                    translation_validated_segments, buffer_translation = await asyncio.to_thread(self.translation.process)
                    async with self.lock:
                        existing = self.state.translation_validated_segments
                        if not isinstance(existing, list):
                            existing = [existing] if existing else []
                        if translation_validated_segments and getattr(translation_validated_segments, "text", ""):
                            existing.append(translation_validated_segments)
                        # Keep memory bounded; UI only needs recent segments
                        self.state.translation_validated_segments = existing[-200:]
                        self.state.buffer_translation = buffer_translation

                    if flush_on_silence:
                        flushed, empty = self.translation.validate_buffer_and_reset()
                        if flushed and getattr(flushed, "text", ""):
                            async with self.lock:
                                existing = self.state.translation_validated_segments
                                if not isinstance(existing, list):
                                    existing = [existing] if existing else []
                                existing.append(flushed)
                                self.state.translation_validated_segments = existing[-200:]
                                self.state.buffer_translation = empty
                self.translation_queue.task_done()
                for _ in additional_tokens:
                    self.translation_queue.task_done()
                
                if sentinel_found:
                    logger.debug("Translation processor received sentinel in batch. Finishing.")
                    await _flush_translation_buffer()
                    break
                
            except Exception as e:
                logger.warning(f"Exception in translation_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'item' in locals() and item is not SENTINEL:
                    self.translation_queue.task_done()
                if 'additional_tokens' in locals():
                    for _ in additional_tokens:
                        self.translation_queue.task_done()
        logger.info("Translation processor task finished.")

    async def sentence_processor(self):
        """Delegate to :class:`~whisperlivekit.sentence_detector.SentenceDetectionProcessor`."""
        await self._sentence_proc.run()

    async def offline_translation_processor(self):
        """Delegate to :class:`~whisperlivekit.translategemma.GemmaTranslationProcessor`."""
        await self.translation.run()

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                if self._ffmpeg_error:
                    yield FrontData(status="error", error=f"FFmpeg error: {self._ffmpeg_error}")
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue

                state = await self.get_current_state()
                
                async with self.lock:
                    tokens_snapshot = list(self.state.tokens)

                # Snapshotted before formatting and reused by the termination
                # check below, so the last iteration always formats with
                # final_flush=True and withheld lines are released before exit.
                final_flush = self.is_stopping and self._processing_tasks_done()

                if self.sentence_detector:
                    from whisperlivekit.sentence_detector import format_sentence_lines
                    # With send_pending off, a finalized sentence is sent only
                    # once its translation has arrived, so caption and
                    # translation appear together. On the final iteration
                    # (stream over, all processors done) everything is
                    # released even if a translation never arrived.
                    withhold_untranslated = (
                        not getattr(self.args, "send_pending", True)
                        and self.use_offline_translation
                        and not final_flush
                    )
                    lines, undiarized_text = await asyncio.to_thread(
                        format_sentence_lines, state, self.args,
                        tokens=tokens_snapshot,
                        withhold_untranslated=withhold_untranslated,
                    )
                else:
                    lines, undiarized_text = await asyncio.to_thread(
                        format_output,
                        state,
                        self.current_silence is not None,
                        args=self.args,
                        sep=self.sep,
                        tokens=tokens_snapshot,
                    )
                if lines and lines[-1].speaker == -2:
                    buffer_transcription = Transcript()
                else:
                    buffer_transcription = state.buffer_transcription

                buffer_diarization = ''
                if undiarized_text:
                    buffer_diarization = self.sep.join(undiarized_text)

                    async with self.lock:
                        self.state.end_attributed_speaker = state.end_attributed_speaker

                buffer_translation_text = ''
                if state.buffer_translation:
                    raw_buffer_translation = getattr(state.buffer_translation, 'text', state.buffer_translation)
                    if raw_buffer_translation:
                        buffer_translation_text = raw_buffer_translation.strip()
                
                response_status = "active_transcription"
                if not state.tokens and not buffer_transcription and not buffer_diarization:
                    response_status = "no_audio_detected"
                    lines = []
                elif not lines:
                    lines = [Line(
                        speaker=1,
                        start=state.end_buffer,
                        end=state.end_buffer
                    )]
                
                # send_pending=False means clients only get finalized content:
                # the unvalidated ASR/translation buffers stay server-side.
                # Status and push decisions above still use the real buffers.
                send_pending = getattr(self.args, "send_pending", True)
                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription.text.strip() if send_pending else '',
                    buffer_diarization=buffer_diarization,
                    buffer_translation=buffer_translation_text if send_pending else '',
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push and (lines or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    if self.archive_writer and lines:
                        self.archive_writer.ingest_lines(lines)
                    yield response
                    self.last_response_content = response
                if self.archive_writer:
                    self.archive_writer.flush_subtitles_if_due()
                
                if final_flush:
                    logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                    return
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter. Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)
        
    async def create_tasks(self):
        """Create and start processing tasks."""

        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        # If using FFmpeg (non-PCM input), start it and spawn stdout reader
        if not self.is_pcm_input:
            success = await self.ffmpeg_manager.start()
            if not success:
                logger.error("Failed to start FFmpeg manager")
                async def error_generator():
                    yield FrontData(
                        status="error",
                        error="FFmpeg failed to start. Please check that FFmpeg is installed."
                    )
                return error_generator()
            self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
            self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
            processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        if self.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        if self.translation:
            if self.use_offline_translation:
                self.translation_task = asyncio.create_task(self.offline_translation_processor())
            else:
                self.translation_task = asyncio.create_task(self.translation_processor())
            self.all_tasks_for_cleanup.append(self.translation_task)
            processing_tasks_for_watchdog.append(self.translation_task)

        if self.sentence_detector:
            self.sentence_task = asyncio.create_task(self.sentence_processor())
            self.all_tasks_for_cleanup.append(self.sentence_task)
            processing_tasks_for_watchdog.append(self.sentence_task)

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        tasks_remaining = [task for task in tasks_to_monitor if task]
        while True:
            try:
                if not tasks_remaining:
                    logger.info("Watchdog task finishing: all monitored tasks completed.")
                    return

                await asyncio.sleep(10)
                
                for i, task in enumerate(list(tasks_remaining)):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                        tasks_remaining.remove(task)
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")
        self.is_stopping = True
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
            
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        if not self.is_pcm_input and self.ffmpeg_manager:
            try:
                await self.ffmpeg_manager.stop()
                logger.info("FFmpeg manager stopped.")
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg manager: {e}")
        if self.diarization:
            self.diarization.close()
            
        if self.transcription and hasattr(self.transcription, 'close'):
            self.transcription.close()
        if self.archive_writer:
            await self.archive_writer.close()
            
        logger.info("AudioProcessor cleanup complete.")

    def _processing_tasks_done(self):
        """Return True when all active processing tasks have completed."""
        tasks_to_check = [
            self.transcription_task,
            self.diarization_task,
            self.translation_task,
            self.sentence_task,
            self.ffmpeg_reader_task,
        ]
        return all(task.done() for task in tasks_to_check if task)


    async def process_audio(self, message):
        """Process incoming audio data."""

        if not self.state.beg_loop:
            self.state.beg_loop = time()
            self.current_silence = Silence(start=0.0, is_starting=True)

        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True

            if self.is_pcm_input:
                # PCM path: flush remaining buffer and signal end directly.
                if self.pcm_buffer:
                    await self._flush_remaining_pcm()
                if self.transcription_queue:
                    await self.transcription_queue.put(SENTINEL)
            else:
                # Non-PCM (FFmpeg) path: close stdin so FFmpeg finishes
                # processing buffered data and exits naturally.
                if self.ffmpeg_manager:
                    await self.ffmpeg_manager.close_stdin()

            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        if self.is_pcm_input:
            self.pcm_buffer.extend(message)
            await self.handle_pcm_data()
        else:
            if self.archive_writer:
                await self.archive_writer.write_audio_chunk(message)
            if not self.ffmpeg_manager:
                logger.error("FFmpeg manager not initialized for non-PCM input.")
                return
            success = await self.ffmpeg_manager.write_data(message)
            if not success:
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot process audio")
                else:
                    logger.warning("Failed to write audio data to FFmpeg")

    async def handle_pcm_data(self):
        # Without VAC, there's no speech detector to end the initial silence.
        # Clear it on the first audio chunk so audio actually gets enqueued.
        if not self.args.vac and self.current_silence:
            await self._end_silence()

        # Process when enough data
        if len(self.pcm_buffer) < self.bytes_per_sec:
            return

        if len(self.pcm_buffer) > self.max_bytes_per_sec:
            logger.warning(
                f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                f"Consider using a smaller model."
            )

        chunk_size = min(len(self.pcm_buffer), self.max_bytes_per_sec)
        aligned_chunk_size = (chunk_size // self.bytes_per_sample) * self.bytes_per_sample

        if aligned_chunk_size == 0:
            return
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:aligned_chunk_size])
        self.pcm_buffer = self.pcm_buffer[aligned_chunk_size:]

        num_samples = len(pcm_array)
        chunk_sample_start = self.total_pcm_samples
        chunk_sample_end = chunk_sample_start + num_samples

        vad_events = []
        if self.args.vac:
            vad_events = self.vac(pcm_array) or []

        # Iterate over events in chronological order and segment the PCM chunk:
        #   [last_offset, end_offset]   -> active audio (tail of speech)
        #   [end_offset, start_offset]  -> silence (skip)
        #   [start_offset, chunk_end]   -> active audio (start of new speech)
        # This properly handles cases where both end and start events fall into the same chunk.
        last_offset = 0
        for event in vad_events:
            if "start" in event and self.current_silence:
                start_sample = int(event["start"])
                # Clamp the start sample to the current chunk boundaries.
                # This ensures we don't retrospectively end a silence period
                # before the current chunk, preventing negative offsets and
                # ensuring all active audio in the current chunk is captured.
                start_sample_eff = max(chunk_sample_start, min(chunk_sample_end, start_sample))
                start_offset = start_sample_eff - chunk_sample_start
                # If the back-dated start reaches before this chunk, recover
                # the missing onset from the dropped-silence pre-roll.
                missing = start_sample_eff - start_sample
                take = min(missing, len(self._silence_preroll)) if missing > 0 else 0
                # Silence ends where re-enqueued audio resumes (pre-roll included),
                # so audio time + silence time stays consistent with stream time.
                await self._end_silence(at_sample=start_sample_eff - take)
                if take > 0:
                    await self._enqueue_active_audio(self._silence_preroll[-take:])
                self._silence_preroll = np.array([], dtype=np.float32)
                last_offset = start_offset

            if "end" in event and not self.current_silence:
                end_sample = int(event["end"])
                # Clamp the end sample to the current chunk boundaries.
                # This prevents double-counting the VAD delay overlap, ensuring
                # the sum of active audio and silence durations strictly equals
                # the physical stream duration, thereby eliminating timestamp drift.
                end_sample_eff = max(chunk_sample_start, min(chunk_sample_end, end_sample))
                end_offset = end_sample_eff - chunk_sample_start
                # The boundary is confirmed: whatever we withheld during the
                # min_silence wait is silence, not speech. It becomes pre-roll.
                if len(self._holdback):
                    self._silence_preroll = np.concatenate(
                        [self._silence_preroll, self._holdback]
                    )[-self._preroll_max_samples:]
                    self._holdback = np.array([], dtype=np.float32)
                if end_offset > last_offset:
                    await self._emit_active_audio(
                        pcm_array[last_offset:end_offset],
                        chunk_sample_start + last_offset,
                    )
                # With holdback, audio after the (back-dated) speech end never
                # reached the ASR, so the silence really starts there — use the
                # unclamped sample, or the next start's back-dating can land
                # before a clamped-to-now begin and yield negative durations.
                await self._begin_silence(at_sample=min(end_sample, chunk_sample_end))
                last_offset = end_offset

        if not self.current_silence and last_offset < num_samples:
            await self._emit_active_audio(
                pcm_array[last_offset:], chunk_sample_start + last_offset
            )
        elif self.current_silence and last_offset < num_samples:
            self._silence_preroll = np.concatenate(
                [self._silence_preroll, pcm_array[last_offset:]]
            )[-self._preroll_max_samples:]

        self.total_pcm_samples = chunk_sample_end

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)

    async def _emit_active_audio(self, segment, seg_start_sample):
        """Enqueue active audio, withholding the part that falls inside the
        VAD's pending-silence window (speech stopped, min_silence not yet
        elapsed). Withheld audio is flushed if speech resumes, or moved to
        the silence pre-roll if the boundary is confirmed."""
        pending_from = None
        if self.vac is not None and self.vac.triggered and self.vac.temp_end:
            pending_from = int(self.vac.temp_end + self.vac.speech_pad_samples)

        if pending_from is None or pending_from >= seg_start_sample + len(segment):
            # No pending silence reaches this segment: speech is certain.
            if len(self._holdback):
                await self._enqueue_active_audio(self._holdback)
                self._holdback = np.array([], dtype=np.float32)
            await self._enqueue_active_audio(segment)
            return

        cut = max(0, pending_from - seg_start_sample)
        if cut > 0:
            # Speech resumed past the previous holdback, then paused again
            # inside this segment: everything before the new pause is speech.
            if len(self._holdback):
                await self._enqueue_active_audio(self._holdback)
                self._holdback = np.array([], dtype=np.float32)
            await self._enqueue_active_audio(segment[:cut])
        self._holdback = np.concatenate([self._holdback, segment[cut:]])

    async def _flush_remaining_pcm(self):
        """Flush whatever PCM data remains in the buffer, regardless of size threshold."""
        # Withheld pending-silence audio precedes anything left in pcm_buffer.
        if len(self._holdback):
            await self._enqueue_active_audio(self._holdback)
            self._holdback = np.array([], dtype=np.float32)
        if not self.pcm_buffer:
            return
        aligned_size = (len(self.pcm_buffer) // self.bytes_per_sample) * self.bytes_per_sample
        if aligned_size == 0:
            return
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:aligned_size])
        self.pcm_buffer = self.pcm_buffer[aligned_size:]

        # End any active silence so the audio gets enqueued
        if self.current_silence:
            await self._end_silence(at_sample=self.total_pcm_samples)

        await self._enqueue_active_audio(pcm_array)
        self.total_pcm_samples += len(pcm_array)
        logger.info(f"Flushed remaining PCM buffer: {len(pcm_array)} samples ({len(pcm_array)/self.sample_rate:.2f}s)")

    async def _enqueue_active_audio(self, pcm_array):
        if pcm_array is None or pcm_array.size == 0:
            return
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(pcm_array.copy())
            self._enqueued_stream_time += len(pcm_array) / self.sample_rate

        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(pcm_array.copy())
