# Limitations & Future Improvements

Honest accounting of what this system does **not** do well yet, and what would
make it production-grade. The brief values sound reasoning and honest measurement
over impressive-looking figures — this document is written in that spirit.

## Current limitations

**1. No quantitative accuracy yet.** The platform is wired end-to-end and the
rules engine is unit-tested, but behaviour-detection accuracy has not been
measured against labelled ground truth. With a free VLM and ~4-5 short clips,
the demo proves the *plumbing and validation logic*, not detection quality. The
SLO sheet defines the quality metrics (precision/recall) but they are currently
unmeasured (no labelled set).

**2. Single-person assumption.** Face recognition uses the highest-confidence
face, and zone checks use one test point (face centre or frame centre). Scenes
with several people simultaneously are not disambiguated per person. Multi-person
tracking (assigning behaviours and zones to specific tracked individuals) is not
implemented.

**3. Per-frame perception, weak temporal modelling.** The VLM sees one frame plus
a short text summary of recent frames. It cannot watch motion across frames, so
behaviours that only read as motion (e.g. "pacing" vs "standing") are inferred
from single stills and context, not true temporal analysis. Duration estimates
are approximate.

**4. Behaviour duration is sample-interval-grained.** Idle/loitering time is
estimated from inter-sample gaps. At a 5s interval, timing is accurate to ~±5s.
Tighter timing needs a smaller interval (more cost) or motion-based estimation.

**5. No RAG / long-term memory.** The system does not retrieve past-session
history or operator-defined "properly cleaned" standards from a vector store.
For the current scale (3 use cases, small rule sets) this is a deliberate
omission, not a gap — see the trade-off below. It becomes a real limitation once
rules grow large or per-person history matters.

**6. Free-tier VLM constraints.** Gemini free tier is rate-limited (10-15 rpm).
Sustained multi-feed live monitoring will hit limits; that path needs a paid tier
or a self-hosted open VLM. The HuggingFace serverless free route for vision
models no longer exists.

**7. Alert delivery is local only.** Alerts go to logs and a JSONL file. There is
no webhook/email/UI delivery yet (the sink interface exists; the sinks don't).

**8. Face recognition quality.** DeepFace with the default `VGG-Face` model and a
cosine threshold is a baseline. It is sensitive to angle, lighting, and threshold
choice, and is not benchmarked here. `enforce_detection=false` means missed faces
silently fall back to anonymous.

**9. No persistence/database.** Sessions and alerts are flat files. There is no
queryable store, no dedup across restarts, no retention policy.

## Deliberate trade-offs (not accidents)

- **No LangGraph for the rules engine.** A stateful graph framework is overkill
  for an accumulator plus threshold checks. Plain Python is smaller, faster to
  reason about, and equally satisfies "rules declarative and externalised."
- **No LangChain RAG.** Retrieval over a handful of rules solves a problem the
  system doesn't have at this scale. Adding it now would be dependency weight and
  failure surface for no behavioural gain.
- **Threads over async.** Simpler and adequate for the target feed count.

## Future improvements (roughly prioritised)

1. **Collect a labelled eval set** and measure SLIs 9-12 (precision, recall,
   per-behaviour accuracy). Everything else is guesswork until this exists.
2. **Multi-person tracking** (e.g. ByteTrack/DeepSORT) so identity, zone, and
   behaviour attach to specific tracked people.
3. **Short clip windows to the VLM** (a few stacked frames or a montage) for real
   temporal behaviour, replacing single-frame + text-history.
4. **Alert delivery sinks**: webhook, email, and a FastAPI status/alerts endpoint
   (the `AlertSink` interface already supports this — only the sinks are missing).
5. **Persistence**: a real database for sessions/alerts with retention and query.
6. **RAG layer** (only once rule sets or per-person history justify it): index
   use-case standards and session summaries, retrieve top-k into the prompt.
7. **Self-hosted open VLM** (e.g. a quantised vision model on local GPU) to remove
   rate limits and keep data on-prem.
8. **Confidence-gated re-query**: when behaviour confidence is low, re-prompt the
   VLM or escalate, instead of accepting the first answer.
9. **A small monitoring script** that computes the SLI ratios from the logs and
   prints an SLO dashboard.
