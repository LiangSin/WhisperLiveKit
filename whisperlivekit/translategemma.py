"""
Google TranslateGemma integration.

Public surface:
  - TranslateGemmaModel        — thin wrapper around the HF model
  - TranslationDispatcher      — centralized batched queue (shared across connections)
  - GemmaTranslationProcessor  — async pipeline stage (used by AudioProcessor)
"""

import asyncio
import logging
import traceback
import torch
from typing import List, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TranslateGemmaModel:
    """
    Loads and wraps a Infomaniak-AI/vllm-translategemma-*-it model for
    sentence-level translation via vLLM with n-gram speculative decoding.

    Parameters
    ----------
    model_size:
        HuggingFace model variant: "4b", "12b", or "27b".
    src_lang:
        BCP-47 source language code (e.g. "zh").
    tgt_lang:
        BCP-47 target language code (e.g. "en").
    gpu_memory_utilization:
        Fraction of GPU VRAM to allocate. Lower this if Whisper is on the
        same GPU. Raise it if translation is on a dedicated GPU.
    """

    def __init__(self, model_size: str = "4b", src_lang: str = "zh", tgt_lang: str = "en", gpu_memory_utilization: float = 0.65,):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._load_model(model_size, gpu_memory_utilization)

        self.gen_kwargs = SamplingParams(
            temperature=0,
            max_tokens=128,
            stop=["<eos>", "<end_of_turn>"],
        )

    def _load_model(self, model_size: str, gpu_memory_utilization: float):
        model_id = f"Infomaniak-AI/vllm-translategemma-{model_size.lower()}-it"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=512,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 4,
            },
            enforce_eager=True,
        )

    def _build_prompt(self, text: str) -> str:
        """Build a chat-template prompt for a single source text."""
        messages = [{
            "role": "user",
            "content": (
                f"<<<source>>>{self.src_lang}"
                f"<<<target>>>{self.tgt_lang}"
                f"<<<text>>>{text}"
            )
        }]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def translate(self, text: str) -> str:
        """Translate *text* from src_lang to tgt_lang."""
        text = (text or "").strip()
        if not text:
            return ""
        inputs = self._build_prompt(text)
        outputs = self.model.generate([inputs], self.gen_kwargs)
        return outputs[0].outputs[0].text.strip()

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a list of texts in a single batched vLLM call."""
        cleaned = [(t or "").strip() for t in texts]
        results = [""] * len(cleaned)

        non_empty = [(i, t) for i, t in enumerate(cleaned) if t]
        if not non_empty:
            return results

        indices, valid_texts = zip(*non_empty)
        prompts = [self._build_prompt(t) for t in valid_texts]
        outputs = self.model.generate(prompts, self.gen_kwargs)

        for idx, output in zip(indices, outputs):
            results[idx] = output.outputs[0].text.strip()

        return results


# ---------------------------------------------------------------------------
# Centralized batched dispatcher (shared across all connections)
# ---------------------------------------------------------------------------

class TranslationDispatcher:
    """
    Collects translation requests from all connections, batches them, and
    calls translate_batch() once per batch for efficient GPU utilization.

    Parameters
    ----------
    gemma_model:
        A loaded ``TranslateGemmaModel``.
    max_batch_wait:
        Maximum seconds to wait for additional requests before dispatching
        a batch.  Keep low for real-time responsiveness.
    max_batch_size:
        Upper limit on prompts per vLLM generate() call.
    """

    def __init__(
        self,
        gemma_model: TranslateGemmaModel,
        max_batch_wait: float = 0.05,
        max_batch_size: int = 32,
    ):
        self.gemma_model = gemma_model
        self.max_batch_wait = max_batch_wait
        self.max_batch_size = max_batch_size
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()

    async def ensure_started(self):
        """Start the dispatch loop if it is not already running."""
        async with self._start_lock:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._dispatch_loop())

    async def translate(self, text: str) -> str:
        """Submit a single text and await the translation result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def _dispatch_loop(self):
        """Drain the queue, batch requests, translate, distribute results."""
        logger.info("TranslationDispatcher: dispatch loop started.")
        try:
            while True:
                batch_texts: List[str] = []
                batch_futures: List[asyncio.Future] = []

                text, future = await self._queue.get()
                batch_texts.append(text)
                batch_futures.append(future)

                deadline = asyncio.get_event_loop().time() + self.max_batch_wait
                while len(batch_texts) < self.max_batch_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        text, future = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining,
                        )
                        batch_texts.append(text)
                        batch_futures.append(future)
                    except asyncio.TimeoutError:
                        break

                logger.debug(
                    "TranslationDispatcher: dispatching batch of %d request(s).",
                    len(batch_texts),
                )

                try:
                    results = await asyncio.to_thread(
                        self.gemma_model.translate_batch, batch_texts,
                    )
                    for fut, result in zip(batch_futures, results):
                        if not fut.done():
                            fut.set_result(result)
                except Exception as exc:
                    logger.warning("TranslationDispatcher batch error: %s", exc)
                    for fut in batch_futures:
                        if not fut.done():
                            fut.set_exception(exc)

        except asyncio.CancelledError:
            logger.info(
                "TranslationDispatcher: dispatch loop cancelled, "
                "resolving %d pending future(s).", self._queue.qsize(),
            )
            while not self._queue.empty():
                try:
                    _, future = self._queue.get_nowait()
                    if not future.done():
                        future.cancel()
                except asyncio.QueueEmpty:
                    break
            raise


# ---------------------------------------------------------------------------
# Async pipeline stage
# ---------------------------------------------------------------------------

class GemmaTranslationProcessor:
    """
    Per-connection consumer: drains sentences from ``translation_sentence_queue``, 
    submits them to the shared ``TranslationDispatcher``,
    and stores results in ``state.translation_validated_segments``.

    Parameters
    ----------
    dispatcher:
        The shared ``TranslationDispatcher``.
    translation_sentence_queue:
        Queue of (Sentence, is_pending: bool) tuples to translate.
    state:
        Shared `~whisperlivekit.timed_objects.State` for this connection.
    lock:
        asyncio.Lock protecting state.
    sentinel:`
        End-of-stream sentinel object (identity-compared).
    """

    def __init__(
        self,
        dispatcher: TranslationDispatcher,
        translation_sentence_queue: asyncio.Queue,
        state,
        lock: asyncio.Lock,
        sentinel: object,
    ):
        self.dispatcher = dispatcher
        self.translation_sentence_queue = translation_sentence_queue
        self.state = state
        self.lock = lock
        self.sentinel = sentinel

    async def run(self):
        """Main processing loop — run as an asyncio.Task."""
        from whisperlivekit.timed_objects import Translation

        await self.dispatcher.ensure_started()

        while True:
            try:
                item = await self.translation_sentence_queue.get()

                if item is self.sentinel:
                    logger.debug("GemmaTranslationProcessor: sentinel received.")
                    self.translation_sentence_queue.task_done()
                    break

                # Drain available items for local batching
                items = [item]
                while True:
                    try:
                        more = self.translation_sentence_queue.get_nowait()
                        items.append(more)
                        if more is self.sentinel:
                            break
                    except asyncio.QueueEmpty:
                        break

                sentinel_found = False
                sentences = []
                for it in items:
                    if it is self.sentinel:
                        sentinel_found = True
                        break
                    sentence, is_pending = it
                    if is_pending:
                        continue
                    if not (sentence.text or "").strip():
                        continue
                    sentences.append(sentence)

                if sentences:
                    try:
                        results = await asyncio.gather(
                            *[self.dispatcher.translate(s.text) for s in sentences]
                        )
                        translations = [
                            Translation(
                                start=s.start,
                                end=s.end,
                                text=r.strip(),
                            )
                            for s, r in zip(sentences, results)
                        ]
                        async with self.lock:
                            existing = self.state.translation_validated_segments
                            if not isinstance(existing, list):
                                existing = [existing] if existing else []
                            existing.extend(translations)
                            self.state.translation_validated_segments = existing[-200:]
                    except Exception as e:
                        logger.warning(
                            "Gemma translation error for batch of %d: %s",
                            len(sentences), e,
                        )

                for _ in items:
                    self.translation_sentence_queue.task_done()

                if sentinel_found:
                    logger.debug("GemmaTranslationProcessor: sentinel received.")
                    break

            except Exception as e:
                logger.warning("Exception in GemmaTranslationProcessor: %s", e)
                logger.debug("Traceback: %s", traceback.format_exc())
                self.translation_sentence_queue.task_done()

        logger.info("GemmaTranslationProcessor finished.")


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = TranslateGemmaModel(model_size="4b", src_lang="zh", tgt_lang="en")
    print(model.translate("你好，最近怎麼樣？"))
