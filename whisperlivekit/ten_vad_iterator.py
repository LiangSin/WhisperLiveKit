import numpy as np

"""
TEN VAD iterator adapter for WhisperLiveKit.

Wraps the TenVad class (pip install ten-vad) to provide the same streaming
interface as FixedVADIterator from silero_vad_iterator.py:

    vad = TenVADIterator(hop_size=256, threshold=0.5)
    result = vad(pcm_float32_array)   # {'start': N}, {'end': N}, or None
    vad.reset_states()

TEN VAD reference: https://github.com/TEN-framework/ten-vad
"""


class TenVADIterator:
    """
    TEN VAD streaming iterator with the same interface as FixedVADIterator.

    Accepts variable-length float32 audio chunks, buffers internally, and
    emits {'start': sample} / {'end': sample} events using the same state
    machine as Silero VADIterator.

    Parameters
    ----------
    hop_size : int
        Number of samples processed per TEN VAD call. Supported values:
        160 (10 ms) or 256 (16 ms) at 16 kHz. Default: 256.
    threshold : float
        Speech probability threshold in [0, 1]. Probabilities at or above
        this value are treated as speech. Default: 0.5.
    sampling_rate : int
        Audio sampling rate in Hz. TEN VAD only supports 16000. Default: 16000.
    min_silence_duration_ms : int
        Minimum silence duration (ms) required to end a speech segment.
        Default: 100.
    speech_pad_ms : int
        Padding (ms) added to each side of a detected speech segment.
        Default: 30.
    """

    def __init__(
        self,
        hop_size: int = 256,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        if sampling_rate != 16000:
            raise ValueError("TEN VAD only supports 16000 Hz sampling rate.")
        if hop_size not in (160, 256):
            raise ValueError("TEN VAD hop_size must be 160 (10 ms) or 256 (16 ms).")

        self.hop_size = hop_size
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        self._init_vad()
        self.reset_states()

    def _init_vad(self):
        try:
            from ten_vad import TenVad
        except ImportError:
            raise ImportError(
                "ten-vad package not found. Install with:\n"
                "  pip install -U --force-reinstall git+https://github.com/TEN-framework/ten-vad.git\n"
                "or:\n"
                "  pip install ten-vad"
            )
        self._vad = TenVad(self.hop_size, self.threshold)

    def reset_states(self):
        """Reset VAD state. Recreates the TenVad instance (no built-in reset)."""
        self._init_vad()
        self.buffer = np.array([], dtype=np.float32)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @staticmethod
    def _float32_to_int16(x: np.ndarray) -> np.ndarray:
        """Convert normalised float32 [-1, 1] to int16 as required by TEN VAD."""
        clipped = np.clip(x, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def _process_chunk(self, chunk: np.ndarray, return_seconds: bool):
        """
        Run TEN VAD on a single hop_size chunk and advance the state machine.

        Returns {'start': N}, {'end': N}, or None — same as VADIterator.
        """
        int16_chunk = self._float32_to_int16(chunk)
        speech_prob, _ = self._vad.process(int16_chunk)

        self.current_sample += self.hop_size

        # --- state machine (mirrors VADIterator.__call__) ---
        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            speech_start = max(
                0,
                self.current_sample - self.speech_pad_samples - self.hop_size,
            )
            if return_seconds:
                return {"start": round(speech_start / self.sampling_rate, 1)}
            return {"start": int(speech_start)}

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - self.hop_size
            self.temp_end = 0
            self.triggered = False
            if return_seconds:
                return {"end": round(speech_end / self.sampling_rate, 1)}
            return {"end": int(speech_end)}

        return None

    def __call__(self, x: np.ndarray, return_seconds: bool = False):
        """
        Process a variable-length float32 audio chunk.

        Parameters
        ----------
        x : np.ndarray
            Audio samples as float32, shape (N,). Any length is accepted.
        return_seconds : bool
            If True, timestamps are returned in seconds instead of samples.

        Returns
        -------
        dict or None
            {'start': N}  — speech segment started at sample N
            {'end': N}    — speech segment ended at sample N
            None          — no boundary detected in this chunk
        """
        self.buffer = np.append(self.buffer, x)
        ret = None

        while len(self.buffer) >= self.hop_size:
            chunk = self.buffer[: self.hop_size]
            self.buffer = self.buffer[self.hop_size :]

            r = self._process_chunk(chunk, return_seconds)

            if ret is None:
                ret = r
            elif r is not None:
                # Merge: a later 'end' overwrites, a later 'start' after 'end' reopens
                if "end" in r:
                    ret["end"] = r["end"]
                if "start" in r and "end" in ret:
                    del ret["end"]

        return ret if ret != {} else None
