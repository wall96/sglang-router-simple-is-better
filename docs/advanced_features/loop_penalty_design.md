# Loop Detection & Mitigation — Design

## 1. Goal

Stop runaway reasoning loops — generations that enter self-reinforcing
repetition and burn to `max_tokens` instead of finishing — **without** degrading
normal reasoning. The feature is per-request configurable and **off by default**.

Non-goals: fixing the model (this is an inference-time guard), and semantic
understanding of content (we target repetition *structure*, not meaning).

## 2. Requirements — the two loop classes

Loops in production fall into two mechanically distinct classes, and the design
must cover **both**:

| Class | Definition | Families | Example |
|---|---|---|---|
| **Rigid** | the same token span repeats *verbatim* | single-token / short-cycle, separators, URLs, JSON/tool-args, numeric | `- ne - ne`, `==== ====`, `r.jina.ai/http:`, `0,0,0,0` |
| **Templated** | a fixed *prefix* repeats while the *tail varies* | reasoning-marker, generic `prefix: X`, role-marker / turn-hallucination, code-review | `Need maybe maybe {include/use/…}`, `Assistant: User: …`, `Potential bug: {X}` |

The split matters because **verbatim methods (n-gram, DRY) can only see the rigid
class.** The templated class has no long exactly-repeating span, so it requires a
different detector. Both classes produce real runaways (to tens of thousands of
tokens).

## 3. Architecture

```
                        ┌──────────────── per decode step, before sampling ───────────────┐
  output tokens  ──►    │  Rigid detector (DRY)            ─┐                               │
                        │  Templated detector (segment /    ├─►  Persistence gate  ─► Escalation
                        │                     hidden-state) ─┘   (sustained?)         (observe→soft→stop)
                        └────────────────────────────────────────────────────────────────┘
```

| Concern | Mechanism |
|---|---|
| Detect **rigid** loops | **DRY sampler** (verbatim suffix penalty) |
| Detect **templated** loops | **Segment-prefix detector** (v1); **hidden-state probe** (v2, recall upgrade) |
| Suppress benign repeats | **Persistence gate** — only sustained repetition counts |
| Act | **Escalation ladder** — observe → soft (block penalty + EOS bias) → hard force-stop |

All of it runs as a single SGLang logits penalizer, gated off by default.

## 4. Components

### 4.1 Rigid detection — DRY

Penalizes any token that would extend a **verbatim repeated suffix** already in
the context:

```
penalty(next_token) = dry_multiplier · dry_base^(match_len − dry_allowed_length),   match_len ≥ dry_allowed_length
```

- `match_len` = length of the longest earlier verbatim match of the current suffix.
- `dry_sequence_breakers` (newline, sentence punctuation) block matches from crossing natural boundaries, so normal short repeats are untouched.
- Penalty grows exponentially with match length → the longer the verbatim retrace, the harder it is to continue. Crushes rigid loops; leaves normal language alone.

Defaults: `dry_multiplier=0.8`, `dry_base=1.75`, `dry_allowed_length=2`,
`dry_range=0` (whole output).

### 4.2 Templated detection (v1) — segment-prefix

Catches loops where a sentence/line prefix repeats while the tail varies:

1. **Segment** the token stream on breaker tokens (newline + sentence punctuation), derived once from the tokenizer.
2. For each completed segment, take its first `seg_prefix_len` tokens as the segment **prefix** (e.g. `Need maybe maybe`, `Potential bug :`, `Assistant : User`).
3. Over the last `seg_window` segment prefixes, compute the **dominant prefix's share** = `max_count / seg_window`.
4. A segment-close is **hot** when share `≥ seg_frac` and the window is full.

It keys on the **repeated prefix, not a verbatim span**, so `Need maybe maybe
{include/use/ensure}` registers despite varying tails. It is **not** keyed to any
specific phrase, so the one mechanism covers all four templated families.

Defaults: `seg_prefix_len=3`, `seg_window=12`, `seg_frac=0.5`.

### 4.3 Templated detection (v2) — hidden-state probe *(planned)*

Surface-text detection is a **lagging** signal — a templated loop is semantically
present before it is textually obvious, which caps the recall of §4.2. The
higher-ceiling approach is a small linear probe on the model's hidden states that
classifies "loop-prone" vs normal reasoning, with a CUSUM change-point trigger
(per arXiv 2601.05693, which reports loop onset detectable ~1500 tokens before
visible repetition). This is the recall upgrade path; it is a separate workstream
(needs a feasibility study on Nex-N2 hidden states) and is **not** required for v1.

### 4.4 Persistence gate

A legitimate table or URL is locally repetitive but **bounded** and
self-terminates; a real loop is **sustained**. Each detector feeds a
consecutive-hot counter, and intervention only triggers when repetition persists:

- `dry`/rigid path and segment path each maintain a "hot run" counter (consecutive hot steps / hot segment-closes), reset to 0 the instant a non-repetitive step appears.
- **soft** triggers at `rigid_run ≥ persist_soft` **or** `seg_hot_run ≥ seg_soft`.
- **hard** triggers at `rigid_run ≥ persist_hard` **or** `seg_hot_run ≥ seg_hard`.

Defaults: `persist_soft=24`, `persist_hard=64`, `seg_soft=4`, `seg_hard=10`.
(Thresholds are initial values to be tuned against a labeled set — see §7.)

### 4.5 Intervention ladder

Acts on the **whole repeating block**, not a single token (single-token penalties
don't break multi-token loops — the model just shifts by one and continues):

1. **observe** — record detector stats + debug log, change nothing (validation / shadow mode).
2. **soft** — subtract `loop_penalty` from the block-continuation tokens, **plus** an escalating EOS bias `= eos_bias · (1 + over)`, where `over` = steps past the soft gate. The longer it loops, the more strongly termination is favored.
3. **hard (force-stop)** — set all logits to −∞ except EOS, forcing clean termination (fallback: hard-block the continuation tokens if EOS is unknown).

## 5. SGLang integration

- A `_BatchedPenalizer` registered in `BatchedPenalizerOrchestrator`; runs every decode step before sampling.
- Per-request state in one `_ReqLoopState` object → batch filter/merge are a single list reindex.
- **Driven from `req.output_ids`** (synced at the top of `_apply`), *not* the per-token `_cumulate` hook — so it is correct under speculative decoding (`NEXTN`), which may bypass that hook. First sync seeds the last `window`; subsequent syncs ingest the per-step delta.
- Breaker (SEP) token ids and EOS ids are derived once from `req.tokenizer` / `req.eos_token_ids` (memoized); nothing tokenizer-related on the hot path.
- The orchestrator is created once per batch and reused across decode steps → incremental detector state is the correct model.
- **Disabled by default**: when not enabled the penalizer does nothing beyond a cheap `_is_required` check.

## 6. Parameters (per request)

| Param | Default | Meaning |
|---|---|---|
| `loop_enable` | false | master switch |
| `loop_window_size` | 256 | sliding window of recent output tokens |
| `loop_min_output_tokens` | 64 | don't detect before this many output tokens |
| **rigid (DRY)** | | |
| `dry_multiplier` | 0.8 | DRY strength (0 disables rigid detection) |
| `dry_base` | 1.75 | DRY exponential base |
| `dry_allowed_length` | 2 | match length below which DRY does not penalize |
| **templated (segment)** | | |
| `seg_prefix_len` | 3 | tokens taken as a segment prefix |
| `seg_window` | 12 | number of recent segments compared |
| `seg_frac` | 0.5 | dominant-prefix share to count a segment-close as hot |
| **persistence** | | |
| `persist_soft` / `persist_hard` | 24 / 64 | rigid-path gates |
| `seg_soft` / `seg_hard` | 4 / 10 | segment-path gates |
| **intervention** | | |
| `loop_penalty` | 0.0 | soft block-continuation penalty |
| `loop_eos_bias` | 8.0 | base EOS bias when intervening |
| `loop_mode` | `observe` | `observe` / `soft` / `hard` |

## 7. What must be validated before any non-observe rollout

This design is a proposal; the following are **prerequisites**, not afterthoughts:

1. **Recall** of templated detection — the segment signal is a surface/lagging signal; measure its miss rate and decide whether v2 (hidden-state) is required.
2. **False-positive rate on normal traffic** — markdown tables, bullet lists, and indented code all repeat a leading prefix and could trip the segment detector. Must be measured at scale.
3. **Threshold tuning** — all gate thresholds are initial guesses; tune against a labeled set with an independent ground-truth metric.
4. **Intervention efficacy** — confirm soft/hard actually reduce runaway rate (detection ≠ mitigation).
5. **Force-stop safety** — forcing EOS mid-reasoning must produce well-formed output under the qwen3 reasoning-parser and qwen3_coder tool-call-parser.
6. **Spec-decode + intervention** — validate the active penalty/force-stop path (not just observe) under `NEXTN`.

Validation order: (a) clean detection PR table on loop-prone + normal traffic →
(b) DRY online A/B for the rigid class → (c) intervention A/B (observe→soft) once
detection is acceptable → (d) hidden-state feasibility if recall demands it.

## Appendix — references

- DRY sampler: oobabooga PR #5677; vLLM PR #11368.
- vLLM `repetition_detection` / `max_pattern_size`.
- "Circular Reasoning: Self-Reinforcing Loops in LRMs" — arXiv 2601.05693 (hidden-state probe).
- "Solving LLM Repetition in Production" — arXiv 2512.04419; "Breaking the Loop: DoS in LLMs" — arXiv 2503.00416.
- Holtzman 2019, "The Curious Case of Neural Text Degeneration".
