# Architecture

## 1. Goal and shape

The system is a **platform**, not a housekeeping app. It observes video, uses a
Vision-Language Model (VLM) to understand human behaviour at a semantic level,
and validates that behaviour against rules defined entirely in configuration.
Housekeeping is just one configured use case.

Two design invariants drive everything:

- **One engine, two paths.** Anonymous validation and identity-bound rules run
  through the same perception → rules → alert path. Identity is an optional
  enrichment, not a separate code path.
- **Behaviour lives in data, not code.** Behaviours, definitions, zones,
  identities, thresholds, and alert triggers are all YAML. Adding a use case
  that reuses the platform's rule vocabulary is a config edit.

## 2. Data flow

```
                         config/*.yaml  (declarative: feeds, zones, identities, use_cases)
                                 │
        ┌────────────────────────┼─────────────────────────────────────┐
        ▼                        ▼                                      ▼
  ┌───────────┐          ┌──────────────┐                       ┌──────────────┐
  │ Ingestion │  Frame   │  Perception  │   EnrichedFrame       │    Rules     │
  │ (Stage 1-2)│ Event   │ (Stage 3-4)  │   + VLMResult         │  (Stage 6)   │
  │           │ ───────► │              │ ────────────────────► │              │
  │ OpenCV    │          │ face + zone  │                       │ session      │
  │ sampler   │          │ (parallel)   │                       │ accumulator  │
  │ per feed, │          │      │       │                       │ + declarative│
  │ threaded  │          │      ▼       │                       │ rule checks  │
  └───────────┘          │  VLM adapter │                       └──────┬───────┘
                         │  (swappable) │                              │
                         └──────────────┘                              ▼
                                                                ┌──────────────┐
                                                                │   Alerting   │
                                                                │  (Stage 7)   │
                                                                │ Log + JSONL  │
                                                                │ + verdict    │
                                                                └──────────────┘
```

The **unit of work** is one sampled frame, carried as `FrameEvent` →
`EnrichedFrame` → `VLMResult`. Each stage adds to that object and never reaches
backwards. Callbacks connect stages, so any stage can be tested or replaced in
isolation.

## 3. Stage detail

**Ingestion (Stage 1-2).** `FeedConfig` (Pydantic) validates every feed at
startup, so bad config fails loudly before a camera connects. `FrameSampler`
runs one feed in its own thread: connect (RTSP forced to TCP), sample at the
configured interval, save a timestamped JPEG, emit a `FrameEvent`, auto-reconnect
with exponential backoff. **File sources are sampled by video timestamp**
(`CAP_PROP_POS_MSEC`), not wall-clock — so a 15s clip sampled every 5s yields
frames at ~0/5/10/15s on any machine regardless of decode speed, and duration
rules reason in *scene* seconds. `FeedManager` runs all enabled feeds and can
expand a directory of clips into one sampler per file.

**Perception (Stage 3-4).** Face recognition (DeepFace, optional) and zone check
(OpenCV polygon test, optional) run in parallel and enrich the frame with *who*
and *where*. Both degrade gracefully to anonymous mode when DeepFace is absent or
no one is enrolled — which is the correct default for the housekeeping path. The
VLM (Stage 4) is the intelligence: a frame plus an engineered prompt (use-case
behaviours, definitions, person/zone context, last-N-frame history) returns
**structured JSON**, validated into a `VLMResult`. Parse failures return a safe
empty result rather than a false alert.

**VLM swappability.** `BaseVLMAdapter` owns prompt building, retry, and JSON
parsing; each provider implements only `_call_api()`. Providers: `mock`
(deterministic, offline, default), `gemini` (free), `openai`, `anthropic`,
`huggingface`. Selected by `VLM_PROVIDER` env var or `--vlm` — zero code change.

**Rules (Stage 6).** A stateful, config-driven engine. Per session it accumulates
observations into `SessionState`, then evaluates two rule classes:

- *streaming* (every frame): unauthorised zone entry, unknown person in
  restricted zone, suspicious behaviour, extended idle, loitering threshold.
- *completion* (session end): missing required step, visit too short / too long.

A trigger fires only if it is listed in that use case's `alert_triggers` **and**
its condition holds; thresholds come from the use case's `rules` block. Duration
thresholds accept seconds or minutes. Streaming alerts de-duplicate per session.

**Alerting (Stage 7).** Every rule outcome becomes an `Alert` carrying feed,
trigger, person, zone, frame index, snapshot path, severity, and details.
`AlertSink` is an open interface; `LogAlertSink` and `JsonFileSink` ship.
Adding webhook / email / FastAPI / DB sinks needs no engine change. Each session
ends with a verdict JSON (compliant + fired triggers + summary).

## 4. Key trade-offs

- **No LangChain / LangGraph.** The rules engine is ~250 lines of plain Python.
  For three use cases and small rule sets, a config-driven accumulator plus
  threshold checks satisfies "rules declarative and externalised" with far less
  dependency weight and failure surface than a stateful graph framework. RAG is
  unnecessary at this scale. Both are listed as future work in `LIMITATIONS.md`.
- **Threads, not async.** Each feed is I/O-bound (network/decode/model latency);
  a thread per feed is simple, debuggable, and adequate for the target scale.
- **File sampling by video time.** Deterministic demos on short clips, at the
  cost of not modelling real-time pressure for files (correct, since files
  aren't real-time).
- **Parse failure ≠ violation.** API hiccups must not raise compliance alerts;
  they degrade to "no information."

## 5. Extending

- New use case (reusing existing triggers): add a block to `use_cases.yaml`.
- New feed: add to `feeds.yaml`.
- New VLM: subclass `BaseVLMAdapter`, register one line in `vlm_pipeline.py`.
- New alert channel: implement `AlertSink.emit`, add to the engine's sink list.
- New *kind* of rule: add one method to `RulesEngine` and a trigger name to the
  vocabulary (a documented code change, by design).
