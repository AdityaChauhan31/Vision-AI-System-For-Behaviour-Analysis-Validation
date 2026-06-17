# Vision AI Platform — Behaviour Understanding & Validation

A generic, configurable Vision-Language platform that watches video feeds,
reasons about **what people are doing** (not just object detection), validates
that behaviour against **externalised rules**, and raises **alerts** when rules
are met or violated.

The same engine serves both supported paths:

1. **Anonymous behavioural validation** — validate anyone in frame against
   defined behaviours, zones, and rules.
2. **Identity-bound rules** — register people via face recognition and enforce
   identity-specific restrictions.

A new use case is **configuration only** — no code changes. Three use cases ship
configured (housekeeping validation, identity-bound restriction, loitering
detection), plus a short-clip demo variant.

---

## Pipeline (Stages 1–7)

```
raw video
  → [ingestion]      OpenCV sampling → timestamped JPEG frames        (Stage 1-2)
  → [face + zone]    who is this? / where are they? (parallel, optional) (Stage 3)
  → [VLM]            frame + engineered prompt → structured behaviour JSON (Stage 4)
  → [rules engine]   accumulate across frames, evaluate declarative rules   (Stage 6)
  → [alerts]         log + JSONL record + per-session verdict              (Stage 7)
```

Stage 5 (RAG) is intentionally **not** implemented — see `docs/LIMITATIONS.md`
for the reasoning. For three use cases and small rule sets it adds dependency
weight and failure surface without earning its keep.

---

## Quick start (no API key, no footage)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. make a synthetic demo clip
python tools/make_demo_video.py --seconds 18 --out data/demo_housekeeping.mp4

# 2. run the full pipeline with the mock VLM (deterministic, offline)
python main.py --video data/demo_housekeeping.mp4 --use-case housekeeping_demo_short
```

You'll see per-frame behaviour, then a session **VERDICT**. Outputs land in:

- `frames/`            — sampled JPEGs
- `logs/pipeline.log`  — full run log
- `logs/alerts.jsonl`  — one JSON alert record per line
- `logs/sessions/*.json` — per-session compliance verdict

---

## Run with a real, free VLM

Two free, no-credit-card options. **Groq is recommended** — higher rate limits
and fast vision inference.

**Groq (recommended: 30 req/min, 1000/day, vision via Llama 4 Scout):**

```bash
cp .env.example .env
# get a key at https://console.groq.com/keys, set in .env:
#   VLM_PROVIDER=groq
#   GROQ_API_KEY=gsk_...
python main.py --video data/your_clip.mp4 --use-case housekeeping_demo_short --vlm groq
```

**Google Gemini (also free, but tighter — 5-10 req/min on free tier):**

```bash
pip install google-generativeai
# .env: VLM_PROVIDER=gemini, GEMINI_API_KEY=...   (aistudio.google.com/app/apikey)
python main.py --video data/your_clip.mp4 --use-case housekeeping_demo_short --vlm gemini
```

At a 5s sample interval a clip fires ~6 calls in 30s. Gemini's free tier caps at
5/min and will 429; Groq's 30/min handles it comfortably. If you must use Gemini,
raise `--interval` to 15.

> The HuggingFace adapter targets the new Inference Providers router, which is
> **no longer free** for vision models. Use Groq or Gemini for a free run.

---

## Run from config (multiple feeds, no CLI args)

`config/feeds.yaml` declares all sources. The shipped config has one enabled
feed that treats `data/` as a folder of demo clips:

```bash
python main.py            # processes every .mp4 in data/ via feeds.yaml
```

---

## Project layout

```
vision_ai/
├── main.py                 # entry point — wires all stages
├── config/                 # settings.py + 4 declarative YAMLs (the "no-code" surface)
├── ingestion/              # Stage 1-2: feeds, sampling (config-validated, threaded)
├── perception/             # Stage 3-4: face, zone, VLM adapters + pipeline
│   └── vlm/                # swappable VLM backends (mock/gemini/openai/anthropic/hf)
├── rules/                  # Stage 6-7: session state, rules engine, alerts
├── tools/                  # define_zones, enroll_face, make_demo_video
├── tests/                  # ingestion + rules tests (no camera/API needed)
└── docs/                   # architecture, user manual, SLI/SLO, limitations
```

See `docs/USER_MANUAL.md` for operator instructions and
`docs/ARCHITECTURE.md` for design and trade-offs.

---

## Web API + UI

A FastAPI app with a browser UI ships in `api/`. Upload a clip, watch behaviour
results stream in, and get a compliance verdict with alerts.

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload --port 7860     # open http://localhost:7860
```

Endpoints: `GET /api/health`, `GET /api/use-cases`, `POST /api/analyze`
(multipart video upload), `POST /api/analyze-demo`, `GET /api/jobs/{id}` (polled
for streaming results + verdict), `GET /api/frames/{id}/{file}`. Analysis runs as
a background job so long clips don't block the request.

If no VLM key is set the API runs in **mock mode** (so it never hard-fails); set
`VLM_PROVIDER` + the matching key for real analysis.

## Deploy (free, public)

Recommended host: **Hugging Face Spaces (Docker)** — free, 16 GB RAM, secret
storage, stays warm enough for background jobs. Render works as a fallback. Full
step-by-step in `docs/DEPLOYMENT.md`. Quick local container:

```bash
docker build -t vision-ai .
docker run -p 7860:7860 -e VLM_PROVIDER=groq -e GROQ_API_KEY=gsk_xxx vision-ai
```
