"""
TranslateGemma client integration.

The TranslateGemma model itself runs as a standalone vLLM service (see `translate-gemma/` at the repository root). 
This module is the WhisperLiveKit-side client that talks to that service over HTTPS API.

Public surface:
  - TranslateGemmaClient       — async HTTP client for the remote vLLM server
  - GemmaTranslationProcessor  — async pipeline stage (used by AudioProcessor)
"""

import asyncio
import logging
import os
import traceback
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_MODEL_NAME_TEMPLATE = "Infomaniak-AI/vllm-translategemma-{size}-it"

def _resolve_model_name(model_size: str, model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    return _MODEL_NAME_TEMPLATE.format(size=model_size.lower())


# ---------------------------------------------------------------------------
# HTTPS client
# ---------------------------------------------------------------------------

class TranslateGemmaClient:
    """
    Async client for the standalone TranslateGemma (vLLM) service.

    Parameters
    ----------
    model_size:
        ``"4b"``/``"12b"``/``"27b"``. Used only to derive the default
        ``model_name`` of ``Infomaniak-AI/vllm-translategemma-<size>-it``.
    src_lang / tgt_lang:
        BCP-47 language codes embedded in the prompt.
    base_url:
        OpenAI-compatible endpoint (``...//v1``). If unset, falls back to
        the ``TRANSLATEGEMMA_URL`` env var, then to ``https://localhost:8765/v1``.
    model_name:
        Override the model name sent in requests. Defaults to the value
        derived from ``model_size``.
    api_key:
        Bearer token for the remote server (vLLM accepts ``EMPTY`` by default).
    timeout:
        Per-request timeout in seconds.
    max_tokens:
        ``max_tokens`` field of each chat-completion request.
    request_concurrency:
        Soft cap on simultaneous in-flight requests from this client.
        vLLM still does its own continuous batching upstream.
    ssl_verify:
        Whether httpx verifies the TranslateGemma HTTPS certificate.
    """

    def __init__(
        self,
        model_size: str = "4b",
        src_lang: str = "zh",
        tgt_lang: str = "en",
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_tokens: int = 128,
        request_concurrency: int = 64,
        ssl_verify: bool = True,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required to talk to the TranslateGemma service. "
                "Install it with `pip install httpx`."
            ) from exc

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.base_url = base_url.rstrip("/")
        self.model_name = _resolve_model_name(model_size, model_name)
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.ssl_verify = ssl_verify
        self._semaphore = asyncio.Semaphore(request_concurrency)
        self._client = None  # lazily created in the running event loop
        self._client_lock = asyncio.Lock()

        logger.info(
            "TranslateGemmaClient initialised: base_url=%s model=%s",
            self.base_url, self.model_name,
        )

    async def _get_client(self):
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    import httpx
                    self._client = httpx.AsyncClient(
                        timeout=self.timeout,
                        verify=self.ssl_verify,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
        return self._client

    async def aclose(self):
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _build_messages(self, text: str):
        return [{
            "role": "user",
            "content": (
                f"<<<source>>>{self.src_lang}"
                f"<<<target>>>{self.tgt_lang}"
                f"<<<text>>>{text}"
            ),
        }]

    async def translate(self, text: str) -> str:
        """Translate *text* from src_lang to tgt_lang via the remote server."""
        text = (text or "").strip()
        if not text:
            return ""

        client = await self._get_client()
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(text),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stop": ["<eos>", "<end_of_turn>"],
        }

        async with self._semaphore:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload,
            )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate *texts* concurrently. vLLM batches them on the server side."""
        cleaned = [(t or "").strip() for t in texts]
        results = [""] * len(cleaned)

        non_empty = [(i, t) for i, t in enumerate(cleaned) if t]
        if not non_empty:
            return results

        indices, valid_texts = zip(*non_empty)
        translations = await asyncio.gather(
            *(self.translate(t) for t in valid_texts),
            return_exceptions=False,
        )
        for idx, translation in zip(indices, translations):
            results[idx] = translation
        return results


# ---------------------------------------------------------------------------
# Async pipeline stage
# ---------------------------------------------------------------------------

class GemmaTranslationProcessor:
    """
    Per-connection consumer: drains sentences from ``translation_sentence_queue``, 
    submits them to the shared ``TranslateGemmaClient``,
    and stores results in ``state.translation_validated_segments``.

    Parameters
    ----------
    client:
        The shared ``TranslateGemmaClient``.
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
        client: TranslateGemmaClient,
        translation_sentence_queue: asyncio.Queue,
        state,
        lock: asyncio.Lock,
        sentinel: object,
    ):
        self.client = client
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
                            *[self.client.translate(s.text) for s in sentences]
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
    async def _smoke_test():
        client = TranslateGemmaClient(model_size="4b", src_lang="zh", tgt_lang="en", base_url="https://localhost:8765/v1")
        try:
            print(await client.translate("你好，最近怎麼樣？"))
        finally:
            await client.aclose()

    asyncio.run(_smoke_test())
