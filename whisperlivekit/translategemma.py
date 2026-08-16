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
import time
import traceback
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_MODEL_NAME_TEMPLATE = "Infomaniak-AI/vllm-translategemma-{size}-it"

# Prompt-length budget. 
_MAX_MODEL_LEN = int(os.environ.get("TRANSLATEGEMMA_MAX_MODEL_LEN", "512"))
_TEMPLATE_OVERHEAD = 32       # chat template + <<<source/target/text>>> markers
_SAFETY_MARGIN = 48           # absorbs _estimate_tokens underestimation
_PENDING_GROWTH_RESERVE = 96  # a pending sentence keeps growing after its context is frozen


def _estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate: CJK char ~ 1 token, other ~ 3 chars/token."""
    if not text:
        return 0
    cjk = sum(
        1 for ch in text
        if 0x3000 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF or 0xFF00 <= ord(ch) <= 0xFFEF
    )
    return cjk + (len(text) - cjk + 2) // 3

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

    def _build_messages(self, text: str, context_src: str = ""):
        return [{
            "role": "user",
            "content": (
                f"<<<source>>>{self.src_lang}"
                f"<<<target>>>{self.tgt_lang}"
                f"<<<text>>>{context_src}{text}"
            ),
        }]

    async def translate(
        self,
        text: str,
        prefix: Optional[str] = None,
        context_src: str = "",
        context_prefix: str = "",
        joiner: str = " ",
    ) -> str:
        """Translate *text* from src_lang to tgt_lang via the remote server.

        When *prefix* is given, the decoder is forced to continue from it
        (vLLM ``continue_final_message``) and only the continuation is
        returned, without stripping — the caller stitches ``prefix +
        continuation`` and relies on exact character alignment across calls.

        *context_src* / *context_prefix* inject preceding sentences: the
        source context is prepended inside ``<<<text>>>`` and its (already
        validated) translation is forced at the head of the assistant
        message, so the continuation covers only *text* itself. *joiner*
        sits between ``context_prefix`` and ``prefix`` and must reproduce
        the whitespace the model itself emitted at that seam.
        """
        text = (text or "").strip()
        if not text:
            return ""
        # A forced prefix must never end in whitespace (see the comment in
        # promotable_prefix_length); the seam whitespace lives in `joiner`
        # or in the model's own continuation.
        context_prefix = (context_prefix or "").rstrip()

        if context_prefix and prefix:
            forced = context_prefix + joiner + prefix
        else:
            forced = context_prefix or prefix or ""

        client = await self._get_client()
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(text, context_src=context_src or ""),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stop": ["<eos>", "<end_of_turn>"],
        }
        if forced:
            payload["messages"] = payload["messages"] + [
                {"role": "assistant", "content": forced}
            ]
            payload["add_generation_prompt"] = False
            payload["continue_final_message"] = True

        async with self._semaphore:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload,
            )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content if forced else content.strip()

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
# Display-stability helpers (nllw-style local agreement)
# ---------------------------------------------------------------------------

def _is_safe_cut_char(ch: str) -> bool:
    """Whether the frozen prefix may end right after this character.

    Freezing mid-word (latin scripts) risks forcing a broken decoder prefix,
    so we only cut after whitespace, punctuation, or CJK characters.
    """
    if ch.isspace():
        return True
    cp = ord(ch)
    if 0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF:
        return True  # CJK ideographs, kana, fullwidth forms
    return not ch.isalnum()


def promotable_prefix_length(prev_tail: str, new_tail: str) -> int:
    """Local agreement: length of ``new_tail``'s head that may be frozen.

    Two consecutive re-translations must agree character-by-character, and
    the cut is pulled back to the last safe boundary inside the agreed span.
    """
    agreed = 0
    for a, b in zip(prev_tail, new_tail):
        if a != b:
            break
        agreed += 1
    cut = 0
    for i in range(agreed):
        if _is_safe_cut_char(new_tail[i]):
            cut = i + 1
    # Keep trailing whitespace out of the frozen prefix: a forced prefix that
    # ends with a space makes the model emit another one ("The weather  is"),
    # and leaving the space at the head of the tail keeps successive
    # continuations aligned character-by-character.
    while cut > 0 and new_tail[cut - 1].isspace():
        cut -= 1
    return cut


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
    context_sentences:
        Max number of preceding validated sentences injected as translation
        context. 0 disables context entirely (requests are then identical
        to the context-free behavior).
    """

    def __init__(
        self,
        client: TranslateGemmaClient,
        translation_sentence_queue: asyncio.Queue,
        state,
        lock: asyncio.Lock,
        sentinel: object,
        context_sentences: int = 2,
        batch_window: float = 0.0,
    ):
        self.client = client
        self.translation_sentence_queue = translation_sentence_queue
        self.state = state
        self.lock = lock
        self.sentinel = sentinel
        self.context_sentences = context_sentences
        # When > 0, dispatch is held until the next wall-clock multiple of
        # this window (+50 ms grace) so requests from every connection and
        # every server process on the host leave in one synchronized burst —
        # vLLM then serves the whole burst as a single continuous-batching
        # round and the GPU idles between ticks.
        self.batch_window = batch_window
        # Stable-prefix state for the current pending-sentence generation
        # (nllw-style): the frozen head of the provisional translation only
        # ever grows, so the displayed caption does not jump around between
        # re-translations. Keyed by the pending sentence's start time.
        self._pending_gen_start: Optional[float] = None
        self._stable_prefix: str = ""
        self._prev_tail: str = ""
        # (source_text, display_translation) per tick of this generation.
        # When SaT later cuts the pending text into a finalized sentence, the
        # latest snapshot whose source is a prefix of that sentence gives us a
        # translation covering only its content — a free source/target
        # alignment for the commit-time forced prefix (_commit_prefix).
        self._history: List[tuple] = []
        # Preceding-sentence context, frozen per pending generation
        # _ctx_joiner is the seam whitespace the model emitted between
        # the forced context translation and this sentence's translation (None = not yet seen).
        self._ctx_src: str = ""
        self._ctx_prefix: str = ""
        self._ctx_joiner: Optional[str] = None

    def _reset_pending_generation(self, gen_start: Optional[float]) -> None:
        self._pending_gen_start = gen_start
        self._stable_prefix = ""
        self._prev_tail = ""
        self._history = []
        self._ctx_src = ""
        self._ctx_prefix = ""
        self._ctx_joiner = None

    async def _select_context(self, target_text: str, reserve: int = 0) -> tuple:
        """Pick preceding validated sentences that fit the prompt budget.

        Walks completed sentences newest-first and stops at the first one
        without a validated translation, so the context is always contiguous
        and immediately precedes the target. Returns ``(ctx_src, ctx_prefix)``,
        both possibly empty.
        """
        if self.context_sentences <= 0:
            return "", ""
        budget = (
            _MAX_MODEL_LEN - self.client.max_tokens - _TEMPLATE_OVERHEAD
            - _SAFETY_MARGIN - reserve
            - _estimate_tokens((target_text or "").strip())
        )
        if budget <= 0:
            return "", ""
        srcs: List[str] = []
        translations: List[str] = []
        async with self.lock:
            sentences = list(self.state.sentence_segments or [])
            validated = self.state.translation_validated_segments
            if not isinstance(validated, list):
                validated = [validated] if validated else []
            by_start = {t.start: t for t in validated}
        for s in reversed(sentences):
            if len(srcs) >= self.context_sentences:
                break
            t = by_start.get(s.start)
            src = (s.text or "").strip()
            tr = (t.text or "").strip() if t is not None else ""
            if not src or not tr:
                break  # keep the context contiguous — never skip over a gap
            cost = _estimate_tokens(src) + _estimate_tokens(tr)
            if cost > budget:
                break
            budget -= cost
            srcs.append(src)
            translations.append(tr)
        srcs.reverse()
        translations.reverse()
        # zh sentences carry their own 。？！ so the sources concatenate
        # directly; the translations join with plain spaces.
        return "".join(srcs), " ".join(translations).rstrip()

    def _context_overflows(self, target_text: str) -> bool:
        """Whether the frozen context no longer fits next to the grown target."""
        used = (
            _TEMPLATE_OVERHEAD + 16 + self.client.max_tokens
            + _estimate_tokens(self._ctx_src)
            + _estimate_tokens(self._ctx_prefix)
            + _estimate_tokens((target_text or "").strip())
            + _estimate_tokens(self._stable_prefix)
        )
        return used > _MAX_MODEL_LEN

    async def _translate_pending(self, pending_sentence) -> str:
        """Re-translate the pending sentence with a frozen stable prefix.

        Returns the full display text (stable prefix + tentative tail). The
        stable prefix grows by local agreement: the part of the tail on which
        two consecutive re-translations agree gets frozen, and later requests
        force the decoder to continue from it (vLLM ``continue_final_message``),
        so it can never be rewritten.
        """
        if pending_sentence.start != self._pending_gen_start:
            self._reset_pending_generation(pending_sentence.start)
            self._ctx_src, self._ctx_prefix = await self._select_context(
                pending_sentence.text, reserve=_PENDING_GROWTH_RESERVE
            )
        elif self._ctx_src and self._context_overflows(pending_sentence.text):
            # The sentence outgrew the reserve: drop the context and restart
            # the generation (one-time flicker) instead of risking a vLLM
            # prompt-length error. The reset clears the context, and the
            # generation start matching prevents re-selection above.
            self._reset_pending_generation(pending_sentence.start)

        continuation = await self.client.translate(
            pending_sentence.text,
            prefix=self._stable_prefix or None,
            context_src=self._ctx_src,
            context_prefix=self._ctx_prefix,
            joiner=self._ctx_joiner or " ",
        )
        if self._ctx_prefix and not self._stable_prefix:
            # First tick(s) under a forced context: the model emits its own
            # seam whitespace before this sentence's translation. Freeze it
            # as the joiner for all later requests of this generation (incl.
            # commit) and keep processor-local strings free of it, so every
            # existing alignment invariant operates in context-free space.
            lead = len(continuation) - len(continuation.lstrip())
            if self._ctx_joiner is None:
                self._ctx_joiner = continuation[:lead] or " "
            continuation = continuation[lead:]
        cut = promotable_prefix_length(self._prev_tail, continuation)
        if cut:
            self._stable_prefix += continuation[:cut]
            continuation = continuation[cut:]
        self._prev_tail = continuation
        display = (self._stable_prefix + continuation).strip()
        self._history.append(((pending_sentence.text or "").strip(), display))
        if len(self._history) > 64:
            self._history = self._history[-64:]
        return display

    async def _translate_finalized(
        self, text: str, prefix: str, ctx_src: str, ctx_prefix: str, joiner: str
    ) -> str:
        """Translate a finalized sentence, with a context-free retry.

        A forced context translation can occasionally make the model end the
        turn immediately (it believes the job is done); in that case fall
        back to a plain request so the sentence never loses its translation.
        """
        r = await self.client.translate(
            text,
            prefix=prefix or None,
            context_src=ctx_src,
            context_prefix=ctx_prefix,
            joiner=joiner,
        )
        if ctx_prefix and not (prefix + r).strip():
            r = await self.client.translate(text, prefix=prefix or None)
        return r

    def _commit_prefix(self, sentence_text: str) -> tuple:
        """Forced decoder prefix for a sentence just cut out of the pending text.

        The pending snapshots grew monotonically, so the latest snapshot whose
        source is a prefix of the finalized sentence yields a translation that
        covers only this sentence's content. Its overlap with the frozen
        stable prefix (cut at a safe boundary) is text that was both displayed
        as frozen and belongs to this sentence — forcing it keeps the
        validated translation identical to what the viewer already read.

        Returns ``(prefix, snapshot_translation)``; the snapshot lets the
        caller clamp the on-screen provisional to this sentence's content.
        """
        target = (sentence_text or "").strip()
        if not target or not self._stable_prefix:
            return "", ""
        best = ""
        for src, translation in self._history:
            if src and target.startswith(src):
                best = translation
        if not best:
            return "", ""
        cut = promotable_prefix_length(self._stable_prefix, best)
        return best[:cut], best

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

                if self.batch_window > 0:
                    # Hold until just past the next shared wall-clock tick
                    # (+50 ms so snapshots enqueued exactly on the tick are
                    # included in this round's drain).
                    w = self.batch_window
                    await asyncio.sleep(w - ((time.time() - 0.05) % w))

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
                pending_sentence = None
                for it in items:
                    if it is self.sentinel:
                        sentinel_found = True
                        break
                    sentence, is_pending = it
                    if not (sentence.text or "").strip():
                        continue
                    if is_pending:
                        # Later snapshots supersede earlier ones within a batch.
                        pending_sentence = sentence
                    else:
                        sentences.append(sentence)
                        # A finalized sentence enqueued after a pending snapshot
                        # consumed that snapshot's text — the snapshot is stale.
                        pending_sentence = None

                if sentences or pending_sentence:
                    tasks = []
                    commit_prefixes = []
                    for s in sentences:
                        prefix = ""
                        if s.start == self._pending_gen_start:
                            # This sentence was cut out of the pending text we
                            # have been provisionally translating: force the
                            # already-displayed frozen prefix so the validated
                            # translation does not jump. The remainder starts
                            # a fresh generation.
                            ctx_src, ctx_prefix = self._ctx_src, self._ctx_prefix
                            ctx_joiner = self._ctx_joiner or " "
                            prefix, snapshot = self._commit_prefix(s.text)
                            self._reset_pending_generation(None)
                            if snapshot:
                                # While the validated translation is in flight,
                                # the carried-over provisional still displays
                                # the whole old pending — including content of
                                # the *next* sentence. Clamp it to this
                                # sentence's snapshot so the frozen region only
                                # ever extends, never shrinks, at commit.
                                async with self.lock:
                                    tp = self.state.translation_pending
                                    if tp is not None and tp.start == s.start:
                                        tp.text = snapshot
                                        tp.stable_chars = min(len(prefix), len(snapshot))
                        else:
                            # Finalized without a prior pending generation:
                            # pick context fresh.
                            ctx_src, ctx_prefix = await self._select_context(s.text)
                            ctx_joiner = " "
                        commit_prefixes.append(prefix)
                        tasks.append(self._translate_finalized(
                            s.text, prefix, ctx_src, ctx_prefix, ctx_joiner
                        ))
                    if pending_sentence:
                        tasks.append(self._translate_pending(pending_sentence))
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    translations = []
                    for s, prefix, r in zip(sentences, commit_prefixes, results[:len(sentences)]):
                        if isinstance(r, BaseException):
                            logger.warning(
                                "Gemma translation error for sentence %r: %s",
                                s.text[:60], r,
                            )
                            continue
                        text = (prefix + r).strip() if prefix else r.strip()
                        translations.append(
                            Translation(start=s.start, end=s.end, text=text)
                        )

                    pending_translation = None
                    if pending_sentence:
                        r = results[-1]
                        if isinstance(r, BaseException):
                            # Stable-prefix state is only mutated on success;
                            # the next tick simply retries.
                            logger.warning("Gemma pending translation error: %s", r)
                        elif r:
                            pending_translation = Translation(
                                start=pending_sentence.start,
                                end=pending_sentence.end,
                                text=r,
                                stable_chars=min(len(self._stable_prefix), len(r)),
                            )

                    async with self.lock:
                        if translations:
                            existing = self.state.translation_validated_segments
                            if not isinstance(existing, list):
                                existing = [existing] if existing else []
                            existing.extend(translations)
                            # Keep in sync with State.sentence_segments maxlen: translations
                            # can only attach to sentences still in that window.
                            self.state.translation_validated_segments = existing[-50:]
                            # Any provisional translation predates these
                            # finalized sentences (queue order) — the validated
                            # translation replaces it atomically here.
                            self.state.translation_pending = None
                        if pending_translation is not None:
                            self.state.translation_pending = pending_translation

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
        client = TranslateGemmaClient(
            model_size="4b", src_lang="zh", tgt_lang="en",
            base_url="https://localhost:8765/v1",
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            ssl_verify=False,
        )
        try:
            print("plain:", await client.translate("你好，最近怎麼樣？"))

            # Context A/B: an ambiguous target sentence whose pronoun and
            # terminology only resolve with the preceding sentence. The
            # context translation is forced as the assistant prefix, so the
            # returned continuation must cover ONLY the target sentence —
            # check that the context translation is not echoed/paraphrased.
            ctx_src = "我們今天講矩陣的性質。"
            target = "它的秩是多少？"
            without_ctx = await client.translate(target)
            ctx_translation = await client.translate(ctx_src)
            with_ctx = await client.translate(
                target, context_src=ctx_src, context_prefix=ctx_translation
            )
            print("context source     :", ctx_src)
            print("context translation:", ctx_translation)
            print("target             :", target)
            print("without context    :", without_ctx)
            print("with context       :", repr(with_ctx))
        finally:
            await client.aclose()

    asyncio.run(_smoke_test())
