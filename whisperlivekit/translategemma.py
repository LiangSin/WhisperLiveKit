"""
Google TranslateGemma integration.

Public surface:
  - TranslateGemmaModel        — thin wrapper around the HF model
  - GemmaTranslationProcessor  — async pipeline stage (used by AudioProcessor)
"""

import asyncio
import logging
import traceback
import torch
from typing import Optional

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


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

    def __init__(self, model_size: str = "4b", src_lang: str = "zh", tgt_lang: str = "en", gpu_memory_utilization: float = 0.7,):
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
            enforce_eager=False,
        )

    def translate(self, text: str) -> str:
        """Translate *text* from src_lang to tgt_lang."""
        text = (text or "").strip()
        if not text:
            return ""

        messages = [{
            "role": "user",
            "content": (
                f"<<<source>>>{self.src_lang}"
                f"<<<target>>>{self.tgt_lang}"
                f"<<<text>>>{text}"
            )
        }]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.model.generate(
            [inputs],
            self.gen_kwargs,
        )
        return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Async pipeline stage
# ---------------------------------------------------------------------------

class GemmaTranslationProcessor:
    """
    Translate sentences arrived via translation_sentence_queue and store the translation in state.translation_validated_segments.

    Parameters
    ----------
    gemma_model:
        A loaded `TranslateGemmaModel`.
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
        gemma_model: TranslateGemmaModel,
        translation_sentence_queue: asyncio.Queue,
        state,
        lock: asyncio.Lock,
        sentinel: object,
    ):
        self.gemma_model = gemma_model
        self.translation_sentence_queue = translation_sentence_queue
        self.state = state
        self.lock = lock
        self.sentinel = sentinel

    async def run(self):
        """Main processing loop — run as an asyncio.Task."""
        from whisperlivekit.timed_objects import Translation

        while True:
            try:
                item = await self.translation_sentence_queue.get()

                if item is self.sentinel:
                    logger.debug("GemmaTranslationProcessor: sentinel received.")
                    self.translation_sentence_queue.task_done()
                    break

                sentence, is_pending = item

                if is_pending:
                    # Ignore pending sentences for now. Update in the future.
                    continue

                if not (sentence.text or "").strip():
                    self.translation_sentence_queue.task_done()
                    continue

                try:
                    translation_text = await asyncio.to_thread(
                        self.gemma_model.translate, sentence.text
                    )
                    translated = Translation(
                        start=sentence.start,
                        end=sentence.end,
                        text=translation_text.strip(),
                    )
                    async with self.lock:
                        existing = self.state.translation_validated_segments
                        if not isinstance(existing, list):
                            existing = [existing] if existing else []
                        existing.append(translated)
                        self.state.translation_validated_segments = existing[-200:]
                except Exception as e:
                    logger.warning(
                        "Gemma translation error for %r: %s",
                        sentence.text[:60], e,
                    )

                self.translation_sentence_queue.task_done()

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
