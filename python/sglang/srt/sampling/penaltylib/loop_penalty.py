import logging
from collections import Counter, deque
from numbers import Number
from typing import Dict, List, Optional, Set

import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer

logger = logging.getLogger(__name__)

# Module-level cache of segment-boundary token ids, keyed by id(tokenizer).
# Computing it scans the vocab once; reuse keeps it off the hot path.
_SEP_TOKEN_CACHE: Dict[int, Set[int]] = {}
_SEP_CHARS = set("\n.!?。！？")


def _compute_sep_token_ids(tokenizer) -> Set[int]:
    """Token ids whose surface form contains a sentence/line boundary char.

    Used to segment the token stream so the templated-loop signal can compare
    consecutive segment prefixes (e.g. repeated 'Need maybe maybe ...').
    """
    if tokenizer is None:
        return set()
    key = id(tokenizer)
    cached = _SEP_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    ids: Set[int] = set()
    try:
        vocab = tokenizer.get_vocab()  # {token_str: id}
        for tok, tid in vocab.items():
            s = (
                tok.replace("Ġ", " ")
                .replace("▁", " ")
                .replace("Ċ", "\n")
                .replace("ċ", "\n")
            )
            if any(c in s for c in _SEP_CHARS):
                ids.add(int(tid))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("LoopPenalizer: failed to compute SEP token ids: %r", e)
        ids = set()
    _SEP_TOKEN_CACHE[key] = ids
    return ids


def _eos_token_ids(req) -> Set[int]:
    ids: Set[int] = set()
    e = getattr(req, "eos_token_ids", None)
    if e:
        ids |= {int(x) for x in e}
    tok = getattr(req, "tokenizer", None)
    eid = getattr(tok, "eos_token_id", None) if tok is not None else None
    if eid is not None:
        ids.add(int(eid))
    return ids


class _ReqLoopState:
    """Per-request detector state. One object per request keeps filter/merge
    to a single list reindex instead of a dozen parallel arrays."""

    __slots__ = (
        # config
        "enabled", "ngram", "window", "penalty", "coverage_thresh",
        "gram_min_count", "persist_soft", "persist_hard", "min_output_tokens",
        "hard_block", "observe_only", "debug", "eos_bias",
        "seg_m", "seg_window_size", "seg_frac", "seg_soft", "seg_hard",
        "sep_ids", "eos_ids",
        # rigid (coverage) signal state
        "win", "gram_counts", "repeated_mass", "loop_run",
        # templated (segment) signal state
        "cur_seg", "seg_prefixes", "seg_pref_counts", "seg_hot_run",
        # bookkeeping
        "output_len", "hit_count", "max_loop_run", "max_seg_run",
        "first_soft_len", "first_hard_len", "first_via",
    )

    def __init__(self, req, p):
        sp = req.sampling_params
        self.ngram = _int(sp, "loop_ngram_size", 0)
        self.enabled = self.ngram > 1
        self.window = max(self.ngram, _int(sp, "loop_window_size", 256))
        self.penalty = _float(sp, "loop_penalty", 0.0)
        self.coverage_thresh = _float(sp, "loop_coverage_thresh", 0.5)
        self.gram_min_count = _int(sp, "loop_gram_min_count", 3)
        self.persist_soft = _int(sp, "loop_persist_soft", 24)
        self.persist_hard = _int(sp, "loop_persist_hard", 64)
        self.min_output_tokens = _int(sp, "loop_min_output_tokens", 64)
        self.hard_block = _bool(sp, "loop_hard_block", False)
        self.observe_only = _bool(sp, "loop_observe_only", False)
        self.debug = _bool(sp, "loop_debug", False)
        self.eos_bias = _float(sp, "loop_eos_bias", 8.0)
        # templated-loop (segment) params
        self.seg_m = _int(sp, "loop_seg_prefix_len", 3)
        self.seg_window_size = _int(sp, "loop_seg_window", 12)
        self.seg_frac = _float(sp, "loop_seg_frac", 0.5)
        self.seg_soft = _int(sp, "loop_seg_soft", 4)
        self.seg_hard = _int(sp, "loop_seg_hard", 10)
        # model-derived ids
        self.sep_ids = _compute_sep_token_ids(getattr(req, "tokenizer", None))
        self.eos_ids = _eos_token_ids(req)

        # dynamic state
        self.win: deque = deque()
        self.gram_counts: Counter = Counter()
        self.repeated_mass = 0
        self.loop_run = 0
        self.cur_seg: List[int] = []
        self.seg_prefixes: deque = deque()
        self.seg_pref_counts: Counter = Counter()
        self.seg_hot_run = 0
        self.output_len = 0
        self.hit_count = 0
        self.max_loop_run = 0
        self.max_seg_run = 0
        self.first_soft_len: Optional[int] = None
        self.first_hard_len: Optional[int] = None
        self.first_via: Optional[str] = None

        # State is driven from req.output_ids in _apply (robust under speculative
        # decoding, where the per-token _cumulate hook may be bypassed). Nothing
        # to seed here -- the first _apply syncs from the request's output tail.

    def sync_from_output(self, output_ids) -> None:
        """Catch up structural buffers to the request's generated output.

        Driven from req.output_ids rather than the _cumulate hook so detection
        works identically with and without speculative decoding. The first call
        seeds only the last `window` tokens; subsequent calls ingest the small
        per-step delta.
        """
        n = len(output_ids) if output_ids else 0
        if n <= self.output_len:
            return
        start = self.output_len
        if start == 0 and n > self.window:
            start = n - self.window
            self.output_len = start
        for t in output_ids[start:n]:
            self._ingest(int(t))

    # ------- incremental ingestion (one token) ------- #
    def _bump_gram(self, gram, delta):
        c = self.gram_counts[gram]
        nc = c + delta
        T = self.gram_min_count
        if delta > 0:
            if nc == T:
                self.repeated_mass += T
            elif nc > T:
                self.repeated_mass += 1
        else:
            if c == T:
                self.repeated_mass -= T
            elif c > T:
                self.repeated_mass -= 1
        if nc == 0:
            del self.gram_counts[gram]
        else:
            self.gram_counts[gram] = nc

    def _ingest(self, tok: int):
        """Update structural buffers for one output token (no hit stamping)."""
        self.output_len += 1
        # segment-prefix buffer
        if self.sep_ids and tok in self.sep_ids:
            if len(self.cur_seg) >= self.seg_m:
                prefix = tuple(self.cur_seg[: self.seg_m])
                self.seg_prefixes.append(prefix)
                self.seg_pref_counts[prefix] += 1
                if len(self.seg_prefixes) > self.seg_window_size:
                    old = self.seg_prefixes.popleft()
                    self.seg_pref_counts[old] -= 1
                    if self.seg_pref_counts[old] == 0:
                        del self.seg_pref_counts[old]
                dom = max(self.seg_pref_counts.values())
                hot = (
                    len(self.seg_prefixes) >= self.seg_window_size
                    and dom / len(self.seg_prefixes) >= self.seg_frac
                )
                self.seg_hot_run = self.seg_hot_run + 1 if hot else 0
                self.max_seg_run = max(self.max_seg_run, self.seg_hot_run)
            self.cur_seg = []
        else:
            self.cur_seg.append(tok)
        # 3-gram coverage buffer
        self.win.append(tok)
        n = self.ngram
        if len(self.win) >= n:
            self._bump_gram(tuple(list(self.win)[-n:]), +1)
        if len(self.win) > self.window:
            old_gram = tuple(list(self.win)[:n])
            self.win.popleft()
            self._bump_gram(old_gram, -1)


def _int(sp, name, default):
    v = getattr(sp, name, default)
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    return int(v) if isinstance(v, Number) else default


def _float(sp, name, default):
    v = getattr(sp, name, default)
    if v is None:
        return default
    return float(v) if isinstance(v, Number) else default


def _bool(sp, name, default):
    v = getattr(sp, name, default)
    if v is None:
        return default
    return bool(v) if isinstance(v, bool) else default


class BatchedLoopPenalizer(_BatchedPenalizer):
    """Detect and suppress runaway reasoning loops.

    Two structural signals (see docs/advanced_features/loop_penalty_design.md):

      - coverage : fraction of the recent n-gram window covered by grams seen
                   >= gram_min_count times. Catches rigid repeats (``==== ...``).
      - segment  : windowed dominant-prefix share over recent sentence segments.
                   Catches templated loops (``Need maybe maybe X``) whose tail
                   token varies and so evade exact n-gram matching.

    Each signal has a persistence gate (a one-off table/URL self-terminates and
    never reaches it). Intervention escalates with persistence: soft block of the
    repeating continuation + EOS bias, then a hard force-stop.
    """

    def _states(self) -> List[_ReqLoopState]:
        return self.states

    def _is_required(self) -> bool:
        for req in self.orchestrator.reqs():
            sp = req.sampling_params
            if _int(sp, "loop_ngram_size", 0) > 1 and (
                _float(sp, "loop_penalty", 0.0) > 0
                or _bool(sp, "loop_observe_only", False)
                or _bool(sp, "loop_hard_block", False)
            ):
                return True
        return False

    def _prepare(self):
        self.states: List[_ReqLoopState] = [
            _ReqLoopState(req, self) for req in self.orchestrator.reqs()
        ]

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # No-op: detection is driven from req.output_ids in _apply (see
        # _ReqLoopState.sync_from_output) so it is robust to speculative decoding
        # bypassing this per-token hook.
        return

    def _stamp(self, st: _ReqLoopState, soft: bool, hard: bool, via: str, req):
        if soft and st.first_soft_len is None:
            st.first_soft_len = st.output_len
            st.first_via = via
        if hard and st.first_hard_len is None:
            st.first_hard_len = st.output_len
        st.hit_count += 1
        req.loop_penalty_stats = {
            "hit_count": st.hit_count,
            "output_len": st.output_len,
            "loop_run": st.loop_run,
            "seg_hot_run": st.seg_hot_run,
            "max_loop_run": st.max_loop_run,
            "max_seg_run": st.max_seg_run,
            "first_soft_len": st.first_soft_len,
            "first_hard_len": st.first_hard_len,
            "via": st.first_via,
            "observe_only": st.observe_only,
            "hard_block": st.hard_block,
        }
        if st.debug and (st.hit_count == 1 or st.hit_count in (2, 4, 8, 16, 32, 64, 128)):
            logger.info(
                "LoopPenalizer hit: rid=%s via=%s soft=%s hard=%s output_len=%s "
                "loop_run=%s seg_hot_run=%s",
                getattr(req, "rid", None), via, soft, hard, st.output_len,
                st.loop_run, st.seg_hot_run,
            )

    def _block_continuation_tokens(self, st: _ReqLoopState) -> List[int]:
        """Tokens that would continue the dominant repeating n-gram."""
        if len(st.win) < st.ngram:
            return []
        suffix = tuple(list(st.win)[-(st.ngram - 1):])
        T = st.gram_min_count
        vocab = self.orchestrator.vocab_size
        return [
            g[-1]
            for g, c in st.gram_counts.items()
            if c >= T and g[: st.ngram - 1] == suffix and 0 <= g[-1] < vocab
        ]

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        reqs = self.orchestrator.reqs()
        for idx, st in enumerate(self.states):
            if not st.enabled:
                continue
            req = reqs[idx] if idx < len(reqs) else None
            if req is not None:
                st.sync_from_output(getattr(req, "output_ids", None))
            if st.output_len < st.min_output_tokens:
                continue

            # coverage (rigid) signal
            total = len(st.win) - (st.ngram - 1)
            cov = st.repeated_mass / total if total > 0 else 0.0
            if cov >= st.coverage_thresh:
                st.loop_run += 1
                st.max_loop_run = max(st.max_loop_run, st.loop_run)
            else:
                st.loop_run = 0

            cov_soft = st.loop_run >= st.persist_soft
            cov_hard = st.loop_run >= st.persist_hard
            seg_soft = st.seg_hot_run >= st.seg_soft
            seg_hard = st.seg_hot_run >= st.seg_hard
            soft = cov_soft or seg_soft
            hard = cov_hard or seg_hard
            if not soft:
                continue

            via = "cov" if cov_soft and not seg_soft else (
                "seg" if seg_soft and not cov_soft else "both")
            if req is not None:
                self._stamp(st, soft, hard, via, req)

            if st.observe_only:
                continue

            # --- intervention: escalate with persistence ---
            if hard and (st.hard_block or st.eos_ids):
                if st.eos_ids:
                    logits[idx, :] = -float("inf")
                    for e in st.eos_ids:
                        logits[idx, e] = 0.0
                    continue
                # no eos known: fall back to blocking the repeating block hard
                block = self._block_continuation_tokens(st)
                if block:
                    logits[idx, block] = -float("inf")
                continue

            # soft: penalize the repeating block + bias toward EOS
            block = self._block_continuation_tokens(st)
            if block and st.penalty > 0:
                logits[idx, block] -= st.penalty
            if st.eos_ids:
                over = max(st.loop_run - st.persist_soft, st.seg_hot_run - st.seg_soft, 0)
                bias = st.eos_bias * (1 + over)
                for e in st.eos_ids:
                    logits[idx, e] += bias
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        idx = keep_indices.detach().to("cpu").tolist()
        self.states = [self.states[i] for i in idx]

    def _merge(self, their: "BatchedLoopPenalizer"):
        self.states.extend(their.states)

    def _teardown(self) -> None:
        if hasattr(self, "states"):
            del self.states
