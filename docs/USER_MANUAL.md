# User Manual (Operator Guide)

This guide is for operators setting up and running the platform. No source
editing is required for any task here.

## 1. Installation

Requirements: Python 3.10+, ~1 GB disk for frames/logs. A GPU is optional
(only relevant if you enable local face recognition).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
python -m pytest tests/ -q         # should report "23 passed"
```

## 2. Choosing a VLM backend

Copy `.env.example` to `.env` and set `VLM_PROVIDER`:

| Provider     | Cost            | Setup                                                   |
|--------------|-----------------|---------------------------------------------------------|
| `mock`       | free, offline   | nothing — default; deterministic fake output for demos  |
| `groq`       | **free, 30 rpm**| key from console.groq.com/keys; vision via Llama 4 Scout — RECOMMENDED |
| `gemini`     | free, 5-10 rpm  | key from aistudio.google.com/app/apikey; `pip install google-generativeai` |
| `openai`     | paid            | `OPENAI_API_KEY`; `pip install openai`                  |
| `anthropic`  | paid            | `ANTHROPIC_API_KEY`; `pip install anthropic`            |
| `huggingface`| paid (router)   | `HUGGINGFACE_API_KEY`; serverless vision no longer free |

Groq is recommended for real runs: 30 requests/min and 1000/day on the free tier
(vs Gemini's 5-10/min), no credit card, and very fast inference.

Check what's configured:

```bash
python -c "from config.settings import settings; settings.print_status()"
```

Keys live only in `.env`, which is gitignored. Never commit keys.

## 3. Running

**A single clip (no config edit):**

```bash
python main.py --video data/clip.mp4 --use-case housekeeping_demo_short --vlm gemini
```

**All configured feeds:**

```bash
python main.py                     # reads config/feeds.yaml
```

Useful flags: `--interval 5` (seconds between samples), `--log-level DEBUG`.

If you have no footage, generate a synthetic clip:

```bash
python tools/make_demo_video.py --seconds 18 --out data/demo_housekeeping.mp4
```

## 4. Configuration (the no-code surface)

All behaviour is controlled by four YAML files in `config/`.

**feeds.yaml** — video sources. Each feed has an `id`, `source_type`
(`rtsp` / `file` / `webcam`), `source`, the `use_case` to apply, sample
interval, and `enabled`. Set `source: "data/"` to treat a folder of clips as
feeds. Toggle `enabled: false` to disable without deleting.

**use_cases.yaml** — the heart. Each use case defines `behaviors_to_detect`,
`behavior_definitions` (fed to the VLM), `required_behaviors`, a `rules` block
of thresholds, and `alert_triggers` (which checks are active). **Add a use case
by adding a block here** — no code change. Supported thresholds:

- `min_duration_seconds` / `max_duration_seconds` (or `_minutes`)
- `all_required_steps_must_complete: true`
- `alert_on_idle_seconds` (or `_minutes`), `idle_behaviors: [...]`
- `max_idle_duration_seconds`, `alert_on_pacing`, `suspicious_behaviors: [...]`

Supported `alert_triggers`: `missing_required_step`, `visit_too_short`,
`visit_too_long`, `extended_idle`, `loitering_threshold_exceeded`,
`suspicious_behavior_detected`, `unauthorized_zone_entry`,
`unknown_person_in_restricted_zone`.

**zones.yaml** — regions per feed. Use `mode: full_frame` when the whole frame
is the area of interest (the simplest setup for single-room clips). Use
`mode: polygon` with pixel coordinates for specific regions; generate
coordinates with `python tools/define_zones.py --video data/clip.mp4`.

**identities.yaml** — enrolled people for the identity-bound path, with allowed
and restricted zones. Not needed for anonymous use cases. Enrol with
`python tools/enroll_face.py --name "Jane" --id staff_001 --source data/jane.jpg`
(requires `pip install deepface`).

## 5. Reading alerts and verdicts

- `logs/pipeline.log` — full run log.
- `logs/alerts.jsonl` — one alert per line. Fields: `trigger`, `severity`
  (`info`/`warning`/`critical`), `message`, `feed_id`, `session_id`,
  `person_id`, `zone_label`, `frame_index`, `snapshot_path`, `details`.
- `logs/sessions/<session_id>.json` — final verdict: `compliant` (bool),
  `fired_triggers`, and a session summary (duration, frames, behaviours seen).
- `frames/` — the sampled JPEGs referenced by `snapshot_path`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Missing API key: GEMINI_API_KEY` | Set it in `.env`, or run with `--vlm mock`. |
| Only 1 frame sampled from a clip | Interval ≥ clip length. Lower `--interval`. |
| `visit_too_short` on a short clip | Expected for short footage; use the `housekeeping_demo_short` use case (second-scale thresholds) for a compliant run. |
| `DeepFace not installed` warning | Harmless for anonymous use cases. Install only if you need identity rules. |
| HuggingFace 404 / auth error | Serverless vision is no longer free; use `--vlm groq`. |
| VLM "parse failed" repeatedly | Output was truncated/invalid. Gemini 2.5 needs JSON mode + token headroom (already set); if it persists, switch to `--vlm groq`. |
| Cannot open RTSP source | Check URL/credentials/network; the feed auto-retries with backoff. |
| Rate limit (429) on Gemini | Free tier is only 5-10 req/min. Raise `--interval`, or switch to `--vlm groq` (30/min). |
| Rate limit (429) on Groq | 30 req/min / 1000 per day. Raise `--interval` or wait; daily resets at midnight UTC. |
