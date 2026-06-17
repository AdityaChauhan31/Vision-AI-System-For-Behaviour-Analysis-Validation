# Deployment Guide

This app ships as a Docker container running FastAPI + a web UI. Below is how to
put it online for free so anyone can use it.

## TL;DR — where to deploy

**Use Hugging Face Spaces (Docker).** It's free, permanent, has 16 GB RAM / 2
vCPU (plenty for OpenCV video work), built-in secret storage for your API key,
and the container stays alive so background analysis jobs run fine. Render's free
tier works too but is RAM-constrained (512 MB) and sleeps aggressively. AWS free
tier is possible but not worth the setup overhead for a free demo.

> Zero-config safety: if no VLM API key is set, the app runs in **mock mode**
> (deterministic demo output) instead of crashing. So it works the moment it
> deploys; add a key to get real analysis.

---

## Option A — Hugging Face Spaces (recommended)

1. Create an account at https://huggingface.co, then **New Space**:
   - **SDK: Docker**  →  *Blank*
   - Name it (e.g. `vision-ai`), set visibility Public.

2. Add this front-matter to the **top of `README.md`** (HF reads `app_port`
   from it). Either edit the existing README or replace it:

   ```
   ---
   title: Vision AI Behaviour Validation
   emoji: 🎥
   colorFrom: blue
   colorTo: purple
   sdk: docker
   app_port: 7860
   ---
   ```

3. Push the project to the Space repo:

   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/vision-ai
   git add .
   git commit -m "Vision AI platform + API + UI"
   git push -u origin main
   ```

4. Add your VLM key as a **Space secret** (Settings → Variables and secrets):
   - `GROQ_API_KEY` = your key from https://console.groq.com/keys
   - `VLM_PROVIDER` = `groq`
   - (optional) `MAX_FRAMES_PER_JOB` = `25`

5. The Space builds the Docker image and serves the UI at
   `https://<your-username>-vision-ai.hf.space`. Share that URL.

Free Spaces sleep after ~48 h idle and wake on the next visit (cold start ~30 s).

---

## Option B — Render (Docker, free web service)

1. Push the project to GitHub.
2. On https://render.com → **New → Web Service** → connect the repo.
   Render auto-detects `render.yaml` (runtime: docker, plan: free).
3. In the dashboard set the secret `GROQ_API_KEY`. `VLM_PROVIDER=groq` is already
   in `render.yaml` (falls back to mock if the key is missing).
4. Deploy. URL looks like `https://vision-ai.onrender.com`.

Caveats: free instances have 512 MB RAM and spin down after 15 min idle (first
request after sleep is slow). Keep clips short and `MAX_FRAMES_PER_JOB` modest.

---

## Option C — any Docker host / your own VM

```bash
docker build -t vision-ai .
docker run -p 7860:7860 -e VLM_PROVIDER=groq -e GROQ_API_KEY=gsk_xxx vision-ai
# open http://localhost:7860
```

`PORT` is read from the environment (default 7860), so the same image runs on
Cloud Run, Fly, a bare VM, etc.

---

## Picking a provider for a public demo

The deployed key is **shared by everyone** who uses the URL, so it draws on one
free quota. Two good free choices, opposite trade-offs:

- **Groq** (`meta-llama/llama-4-scout-17b-16e-instruct`): 30 req/min but only
  6,000 tokens/min — images are token-heavy, so keep the sample interval ≥ 10 s.
- **Gemini** (`gemini-2.5-flash`): 5–10 req/min but 250,000 tokens/min — better
  for images; keep interval ≥ 12 s to stay under the request cap. Needs
  `pip install google-generativeai` (uncomment in requirements).

Protections already built in: per-job frame cap (`MAX_FRAMES_PER_JOB`), upload
size limit (`MAX_UPLOAD_MB`, default 60), and the UI defaults to a 10 s interval.
Tune these via environment variables to control quota burn.

---

## Local development

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload --port 7860
# open http://localhost:7860
```
