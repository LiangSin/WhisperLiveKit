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

from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TranslateGemmaModel:
    """
    Loads and wraps a google/translategemma-*-it model for sentence-level
    translation.

    Parameters
    ----------
    model_size:
        HuggingFace model variant: "4b", "12b", or "27b".
    src_lang:
        BCP-47 source language code (e.g. "zh").
    tgt_lang:
        BCP-47 target language code (e.g. "en").
    """

    def __init__(self, model_size: str = "4b", src_lang: str = "zh", tgt_lang: str = "en"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._load_model(model_size)

        self.gen_kwargs = {"do_sample": False, "max_new_tokens": 128}

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None
        if pad_token_id is None and hasattr(self.model, "generation_config") and self.model.generation_config:
            pad_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if pad_token_id is None and hasattr(self.model, "config") and self.model.config:
            pad_token_id = getattr(self.model.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 1  # Gemma tokenizer default; silences "Setting pad_token_id to eos_token_id" warning

        self.gen_kwargs["pad_token_id"] = pad_token_id

    def _load_model(self, model_size: str = "4b"):
        model_id = f"google/translategemma-{model_size.lower()}-it"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(device)

    def translate(self, text: str) -> str:
        """Translate *text* from src_lang to tgt_lang."""
        text = (text or "").strip()
        if not text:
            return ""

        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.src_lang,
                "target_lang_code": self.tgt_lang,
                "text": text,
            }],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = len(inputs["input_ids"][0])
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True)


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
