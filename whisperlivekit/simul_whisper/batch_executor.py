"""Cross-session batched decode for SimulStreaming (AlignAtt, greedy).

Under multi-session load every decode step is a batch-1 decoder forward
whose cost is dominated by CPU-side dispatch and kernel-launch overhead
(~1.3 ms CPU for cool-whisper's 2-layer decoder), plus repeated weight
reads on the GPU. N sessions issue N of these per step interval from N
threads and the host/GPU timeline saturates long before arithmetic
capacity. This module replaces all sessions' single-token decode steps
with one fixed-shape forward over a slot-resident KV cache that is
captured into a CUDA graph: per step the worker fills three small input
buffers and replays the graph (~0.3 ms CPU regardless of how many
sessions participate).

Architecture:

- KV caches are round-scoped (``AlignAtt._clean_cache`` runs after every
  ``infer()`` round). The round-opening multi-token prefill runs in the
  session's own thread on the legacy dict-cache path (fully concurrent,
  numerically identical), and the resulting K/V is then *adopted* into a
  slot by the worker. ``release()`` (from ``DecoderState.clean_cache``)
  frees the slot.
- Slot KV buffers are laid out ``(layer, slot, head, pos, head_dim)``
  and pre-scaled by ``(head_dim ** -0.25)``, matching the
  ``(q*scale) @ (k*scale)^T`` formulation in
  ``MultiHeadAttention.qkv_attention`` (fp32 softmax path).
- The step forward always runs the full slot width with a fixed KV
  length of ``n_text_ctx``: per-row additive key-padding masks derived
  from a GPU-resident length vector make the extra columns exact no-ops,
  and rows whose session did not submit a request this step ("inactive
  rows") compute garbage that is never observed: their lengths do not
  advance, so the garbage K/V written at ``len`` stays masked, and their
  logits rows are simply not read. Every op is row-independent, so
  garbage cannot leak into participants. Fixed shapes are what make the
  CUDA graph legal; they also cost little (the whole step reads <1 GiB).
- Graph capture happens once at startup; on any capture failure the same
  fixed-shape step runs eagerly (functionally identical, just slower).
- On slot exhaustion a session's round falls back to the legacy dict
  cache; its steps are then executed eagerly by the worker.
"""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _Req:
    __slots__ = ("kind", "state", "tokens", "xa", "fut", "slot", "cache", "t0")

    def __init__(self, kind, state=None, tokens=None, xa=None, fut=None,
                 slot=None, cache=None):
        self.kind = kind
        self.state = state
        self.tokens = tokens
        self.xa = xa
        self.fut = fut
        self.slot = slot
        self.cache = cache
        self.t0 = time.monotonic()


class BatchDecodeExecutor:
    def __init__(self, model, max_slots: int = 16, use_graph: bool = True):
        self.model = model
        dims = model.dims
        self.device = next(model.parameters()).device
        self.n_layer = dims.n_text_layer
        self.n_head = dims.n_text_head
        self.n_state = dims.n_text_state
        self.head_dim = self.n_state // self.n_head
        self.scale = self.head_dim ** -0.25
        self.p_self = dims.n_text_ctx          # 448
        self.p_cross = dims.n_audio_ctx        # 1500
        self.S = max_slots

        dt = torch.float32
        dev = self.device
        shape_self = (self.n_layer, self.S, self.n_head, self.p_self, self.head_dim)
        shape_cross = (self.n_layer, self.S, self.n_head, self.p_cross, self.head_dim)
        self.k_self = torch.zeros(shape_self, device=dev, dtype=dt)
        self.v_self = torch.zeros(shape_self, device=dev, dtype=dt)
        self.k_cross = torch.zeros(shape_cross, device=dev, dtype=dt)
        self.v_cross = torch.zeros(shape_cross, device=dev, dtype=dt)

        # Step inputs (worker fills these, the graph reads them).
        self.in_tokens = torch.zeros((self.S, 1), device=dev, dtype=torch.long)
        self.in_lens = torch.ones(self.S, device=dev, dtype=torch.long)
        self.in_active = torch.zeros(self.S, device=dev, dtype=torch.long)
        self._col = torch.arange(self.p_self, device=dev)
        # Alignment-head indices as device tensors: python-list indexing
        # inside the step would H2D-copy, which CUDA graph capture forbids.
        spec = model.decoder.alignment_spec or {}
        self._spec_idx = {
            l: torch.tensor(heads, device=dev, dtype=torch.long)
            for l, heads in spec.items()
        }

        # Slot table — worker-thread-only after startup.
        self.slot_len = [1] * self.S          # CPU mirror of in_lens
        self.slot_free = list(range(self.S))  # kept sorted, lowest first
        self._exhausted_warned = False

        # Stats (worker-thread-only writes).
        self.stat_steps = 0
        self.stat_step_rows = 0
        self.stat_prefills = 0
        self.stat_fallbacks = 0
        self.stat_wait_s = 0.0
        self.stat_wait_max = 0.0
        self.stat_busy_s = 0.0
        self.stat_adopt_s = 0.0

        # CUDA graph over the fixed-shape step (built on the worker thread
        # at startup so capture and replays share one thread/stream).
        self.use_graph = use_graph and self.device.type == "cuda"
        self._graph = None
        self._graph_out = None    # (logits, {layer: qk})

        gib = sum(t.numel() * t.element_size() for t in
                  (self.k_self, self.v_self, self.k_cross, self.v_cross)) / 2**30
        logger.info("Batched decode enabled: %d slots, %.1f GiB slot KV buffers",
                    self.S, gib)

        # Capture synchronously during engine init, while no other thread
        # is issuing CUDA work (sessions do not exist yet). Replays happen
        # on the worker thread; capture/replay threads may differ.
        if self.use_graph and model.decoder.alignment_spec is None:
            # The graph would bake in "no alignment QK outputs" and starve
            # AlignAtt. The eager path reads the spec per call, so it stays
            # correct if the spec is published later.
            logger.warning(
                "batched decode: alignment_spec not set at init — skipping "
                "CUDA graph capture, running the fixed-shape step eagerly")
            self.use_graph = False
        if self.use_graph:
            try:
                self._build_graph()
                logger.info(
                    "batched decode: CUDA graph captured for the step forward")
            except Exception:
                logger.exception(
                    "batched decode: CUDA graph capture failed — running the "
                    "fixed-shape step eagerly")
                self._graph = None

        self.q = queue.SimpleQueue()
        self.worker = threading.Thread(
            target=self._worker_loop, name="batch-decode", daemon=True)
        self.worker.start()
        # Let the worker drain out before interpreter teardown; a daemon
        # thread killed mid-CUDA-call aborts the process on exit.
        import atexit
        atexit.register(self._shutdown)

    def _shutdown(self):
        self.q.put(_Req("quit"))
        self.worker.join(timeout=3)

    # ---------------------------------------------------------------- API

    def prefill(self, state, tokens, xa):
        """Round-opening multi-token forward.

        Runs in the *calling* (session) thread on the legacy dict-cache
        path — concurrent across sessions and numerically identical to the
        unbatched server — then hands the K/V to the worker for adoption
        into a slot. Returns immediately with the prefill outputs; the
        session's subsequent step() calls are ordered behind the adoption
        by the FIFO request queue.
        """
        cache = {}
        with torch.inference_mode():
            logits, cross_attns = self.model.decoder(
                tokens, xa, kv_cache=cache, return_cross_attn=True)
        state.batch_executor = self
        state.batch_pending = True
        state.kv_cache = cache   # kept for the exhaustion fallback path
        self.q.put(_Req("adopt", state=state, cache=cache))
        return logits, cross_attns

    def step(self, state, tokens):
        fut = Future()
        self.q.put(_Req("step", state=state, tokens=tokens, fut=fut))
        return fut.result()

    def cancel(self, state):
        """End-of-round cleanup. FIFO ordering guarantees this runs after
        the session's pending adoption (if any), so the slot cannot leak
        even when a round ends before its adoption was processed."""
        self.q.put(_Req("release", state=state))

    # ------------------------------------------------------------- worker

    def _worker_loop(self):
        while True:
            first = self.q.get()
            batch = [first]
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if any(r.kind == "quit" for r in batch):
                return
            now = time.monotonic()
            steps, legacy_steps = [], []
            for r in batch:
                if r.kind == "release":
                    self._do_release(r.state)
                elif r.kind == "adopt":
                    t1 = time.monotonic()
                    try:
                        with torch.inference_mode():
                            self._do_adopt(r)
                    except Exception:
                        logger.exception("slot adoption failed — session falls "
                                         "back to unbatched decode")
                        r.state.batch_slot = None
                    self.stat_adopt_s += time.monotonic() - t1
                elif r.kind == "step":
                    w = now - r.t0
                    self.stat_wait_s += w
                    if w > self.stat_wait_max:
                        self.stat_wait_max = w
                    if r.state.batch_slot is not None:
                        steps.append(r)
                    else:
                        legacy_steps.append(r)
            for r in legacy_steps:
                try:
                    with torch.inference_mode():
                        r.fut.set_result(self._do_legacy_step(r))
                except Exception as e:  # noqa: BLE001
                    r.fut.set_exception(e)
            if steps:
                try:
                    t1 = time.monotonic()
                    with torch.inference_mode():
                        self._do_steps(steps)
                    self.stat_busy_s += time.monotonic() - t1
                except Exception as e:  # noqa: BLE001
                    logger.exception("batched step failed")
                    for r in steps:
                        if not r.fut.done():
                            r.fut.set_exception(e)

    def _do_legacy_step(self, r):
        # Exhaustion fallback: this session's round runs on its dict cache.
        # xa is not needed: cross K/V is already in the cache.
        return self.model.decoder(
            r.tokens, self._dummy_xa(), kv_cache=r.state.kv_cache,
            return_cross_attn=True)

    def _dummy_xa(self):
        # TextDecoder.forward only uses xa for dtype and (cached) cross K/V.
        if not hasattr(self, "_xa_stub"):
            self._xa_stub = torch.zeros(
                (1, 1, self.n_state), device=self.device, dtype=torch.float32)
        return self._xa_stub

    def _do_release(self, state):
        slot = state.batch_slot
        state.batch_slot = None
        if slot is None or slot in self.slot_free:
            return
        self.slot_len[slot] = 1
        self.in_lens[slot] = 1
        self.slot_free.append(slot)
        self.slot_free.sort()

    def _do_adopt(self, req):
        state = req.state
        cache = req.cache
        if not self.slot_free:
            if not self._exhausted_warned:
                logger.warning(
                    "batched decode: all %d slots busy — falling back to "
                    "unbatched decode for overflow sessions", self.S)
                self._exhausted_warned = True
            self.stat_fallbacks += 1
            state.batch_slot = None
            return
        slot = self.slot_free.pop(0)
        state.batch_slot = slot
        h, hd = self.n_head, self.head_dim
        dec = self.model.decoder
        cached_len = 1
        for l, block in enumerate(dec.blocks):
            ks = cache[block.attn.key_cache_id]        # (1, T, d)
            vs = cache[block.attn.value_cache_id]
            T = ks.shape[1]
            cached_len = T
            self.k_self[l, slot, :, :T] = (
                ks.view(T, h, hd).permute(1, 0, 2) * self.scale)
            self.v_self[l, slot, :, :T] = vs.view(T, h, hd).permute(1, 0, 2)
            kc = cache[block.cross_attn.key_cache_id]   # (1, 1500, d)
            vc = cache[block.cross_attn.value_cache_id]
            self.k_cross[l, slot] = (
                kc.view(self.p_cross, h, hd).permute(1, 0, 2) * self.scale)
            self.v_cross[l, slot] = vc.view(self.p_cross, h, hd).permute(1, 0, 2)
        self.slot_len[slot] = cached_len
        self.in_lens[slot] = cached_len
        self.stat_prefills += 1

    # ---------------------------------------------------- fixed-shape step

    def _step_fixed(self):
        """One full-width single-token forward over all S slots.

        Reads self.in_tokens/in_lens/in_active; appends K/V and advances
        in_lens for active rows only. Fixed shapes and no host reads —
        CUDA-graph capturable. Returns (logits (S,1,V) fp32,
        {layer: qk (S, n_sel_heads, 1, p_cross) fp32}).
        """
        dec = self.model.decoder
        S, h, hd, d = self.S, self.n_head, self.head_dim, self.n_state
        lens = self.in_lens
        active = self.in_active

        # clamp: a finished row can sit at len == n_text_ctx until released.
        x = dec.token_embedding(self.in_tokens) + \
            dec.positional_embedding[lens.clamp(max=self.p_self - 1)].unsqueeze(1)

        # (S,1,1,P) additive bias: -inf on columns >= len.
        pad_bias = torch.where(
            (self._col[None, :] >= lens[:, None])[:, None, None, :],
            float("-inf"), 0.0)

        # Positions written this step; inactive rows' writes land at their
        # current len, stay masked (len does not advance) and are
        # overwritten whenever the row next really steps. clamp guards the
        # (never-stepping-again) len == p_self edge.
        pos = lens.clamp(max=self.p_self - 1)
        scatter_idx = pos[:, None, None, None].expand(S, h, 1, hd)

        cross_qk = {}

        for l, block in enumerate(dec.blocks):
            hx = block.attn_ln(x)
            q = (block.attn.query(hx) * self.scale).view(S, 1, h, hd).permute(0, 2, 1, 3)
            k_new = (block.attn.key(hx) * self.scale).view(S, 1, h, hd).permute(0, 2, 1, 3)
            v_new = block.attn.value(hx).view(S, 1, h, hd).permute(0, 2, 1, 3)
            self.k_self[l].scatter_(2, scatter_idx, k_new)
            self.v_self[l].scatter_(2, scatter_idx, v_new)
            Kp = self.k_self[l]                       # (S, h, P, hd)
            Vp = self.v_self[l]
            qk_p = torch.matmul(q, Kp.transpose(-1, -2)) + pad_bias
            qk_n = (q * k_new).sum(-1, keepdim=True)   # own token, never masked
            w = F.softmax(torch.cat([qk_p, qk_n], dim=-1), dim=-1)
            out = torch.matmul(w[..., :self.p_self], Vp) + w[..., self.p_self:] * v_new
            x = x + block.attn.out(out.permute(0, 2, 1, 3).reshape(S, 1, d))

            hc = block.cross_attn_ln(x)
            qc = (block.cross_attn.query(hc) * self.scale).view(S, 1, h, hd).permute(0, 2, 1, 3)
            qkc = torch.matmul(qc, self.k_cross[l].transpose(-1, -2))
            wc = F.softmax(qkc, dim=-1)
            outc = torch.matmul(wc, self.v_cross[l])
            x = x + block.cross_attn.out(outc.permute(0, 2, 1, 3).reshape(S, 1, d))
            heads_idx = self._spec_idx.get(l)
            if heads_idx is not None:
                cross_qk[l] = qkc.index_select(1, heads_idx)

            x = x + block.mlp(block.mlp_ln(x))

        x = dec.ln(x)
        logits = (x @ dec.token_embedding.weight.transpose(0, 1)).float()

        self.in_lens.add_(active)
        return logits, cross_qk

    def _build_graph(self):
        torch.cuda.synchronize(self.device)
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self._step_fixed()
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            self._graph_out = self._step_fixed()
        self._graph = g
        # Warmup/capture advanced lens and wrote garbage K/V; reset.
        self.in_lens.fill_(1)
        self.in_active.zero_()
        self.slot_len = [1] * self.S

    def _do_steps(self, reqs: List[_Req]):
        # Drop requests whose slot vanished (session cleaned up mid-round).
        live = [r for r in reqs if r.state.batch_slot is not None]
        for r in reqs:
            if r.state.batch_slot is None:
                r.fut.set_exception(RuntimeError("batch slot released mid-round"))
        if not live:
            return
        part_slots = [r.state.batch_slot for r in live]
        part_idx = torch.tensor(part_slots, device=self.device)

        self.in_active.zero_()
        self.in_active.index_fill_(0, part_idx, 1)
        self.in_tokens.index_copy_(
            0, part_idx, torch.cat([r.tokens for r in live], dim=0))

        if self._graph is not None:
            self._graph.replay()
            logits, cross_qk = self._graph_out
        else:
            logits, cross_qk = self._step_fixed()

        for s in part_slots:
            self.slot_len[s] += 1

        self.stat_steps += 1
        self.stat_step_rows += len(live)
        if self.stat_steps == 100 or self.stat_steps % 2000 == 0:
            logger.info(
                "batched decode: %d steps, avg batch %.2f, %d prefills, %d fallbacks | "
                "wait avg %.1fms max %.0fms | step busy avg %.2fms | adopt avg %.1fms",
                self.stat_steps, self.stat_step_rows / self.stat_steps,
                self.stat_prefills, self.stat_fallbacks,
                1e3 * self.stat_wait_s / max(1, self.stat_step_rows),
                1e3 * self.stat_wait_max,
                1e3 * self.stat_busy_s / self.stat_steps,
                1e3 * self.stat_adopt_s / max(1, self.stat_prefills))
            self.stat_wait_max = 0.0

        n_layer = self.n_layer
        for r, slot in zip(live, part_slots):
            # Clone: graph replays refill the same output buffers.
            attns: List[Optional[torch.Tensor]] = [None] * n_layer
            for l, t in cross_qk.items():
                attns[l] = t[slot:slot + 1].clone()
            r.fut.set_result((logits[slot:slot + 1].clone(), attns))
