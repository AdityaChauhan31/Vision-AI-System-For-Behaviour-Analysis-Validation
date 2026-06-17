# SLI / SLO Sheet

Service Level Indicators (what we measure) and Objectives (the targets) for the
platform. Targets are stated for the demo/reference deployment; tighten them for
production once a real feed and labelled data exist.

## Definitions

- **SLI** — a measured signal (a ratio or latency the system actually emits).
- **SLO** — the target value for that signal over a stated window.
- **Window** — rolling 7 days unless noted.

## Reliability SLIs/SLOs

| # | SLI (what is measured) | How it's measured | SLO (target) |
|---|------------------------|-------------------|--------------|
| 1 | **Frame ingestion success** = frames sampled / frames expected | `frame_count` vs `duration / interval` per session | ≥ 99% for files; ≥ 95% for live feeds |
| 2 | **VLM parse success** = `parse_success=true` results / total results | `VLMResult.parse_success` | ≥ 98% (mock), ≥ 90% (live VLM) |
| 3 | **VLM call success** = non-error responses / attempts (after retry) | adapter retry loop outcome | ≥ 99% |
| 4 | **End-to-end frame latency** (capture → VLMResult) | `VLMResult.latency_ms` + `enrichment_ms` | p95 ≤ 4000 ms (Gemini Flash); p95 ≤ 500 ms (mock) |
| 5 | **Stage-3 enrichment latency** (face + zone) | `EnrichedFrame.enrichment_ms` | p95 ≤ 800 ms with DeepFace; ≤ 20 ms anonymous |
| 6 | **Session finalisation rate** = sessions with a written verdict / sessions started | `logs/sessions/*.json` count vs sessions | 100% |
| 7 | **Feed availability** = time a feed is connected / total runtime | sampler connect/disconnect log | ≥ 99% (files); ≥ 98% (RTSP) |
| 8 | **Reconnect recovery** = unexpected disconnects auto-recovered / total | sampler reconnect outcomes | ≥ 95% within 5 attempts |

## Quality SLIs/SLOs (require labelled data)

These measure *correctness* of validation and need a ground-truth set; they are
defined now and become measurable once a labelled eval set is collected.

| # | SLI | Target |
|---|-----|--------|
| 9  | **Alert precision** = true alerts / all alerts raised | ≥ 0.85 |
| 10 | **Alert recall** = true alerts / all real violations | ≥ 0.80 |
| 11 | **Required-step detection accuracy** (per behaviour) | ≥ 0.85 |
| 12 | **False-alert rate** = false alerts / compliant sessions | ≤ 0.10 |

## Error budget

For a 99% availability SLO over 7 days, the error budget is ~100 minutes of
downtime/week. Reconnect-with-backoff and graceful VLM parse-failure handling are
the mechanisms that protect this budget: a transient camera drop or a malformed
VLM response consumes budget but does not crash the pipeline or emit false alerts.

## Capacity notes (free tiers)

- **Groq (recommended)**: `llama-4-scout` at 30 req/min, 1000 req/day free —
  comfortably handles single-feed short-clip demos and light live use.
- **Gemini**: 5-10 req/min on the free tier; at a 5s interval this 429s within a
  minute. Raise the interval or prefer Groq for live work.
- Sampling interval is the primary cost/coverage dial: SLI #1 (coverage) trades
  directly against request volume (cost and rate-limit headroom).

## How to collect these today

SLIs 1-8 are already emitted in `logs/pipeline.log`, `logs/alerts.jsonl`, and
`logs/sessions/*.json`. A small log-aggregation script (future work) can compute
the ratios per window. SLIs 9-12 require a labelled eval set.
