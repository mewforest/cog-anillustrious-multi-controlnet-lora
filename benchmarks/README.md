# Modal vs Replicate benchmark

Compares cold-start latency (and, once rates are confirmed, cost) between the
Modal deployment (`modal_app.py`) and the Replicate/Cog deployment, both
running the same `predict.py`. Built to add other services later without
reshaping the harness -- see "Adding a service" below.

## Setup

```bash
pip install -r benchmarks/requirements.txt
cp benchmarks/.env.example benchmarks/.env
# edit benchmarks/.env: REPLICATE_API_TOKEN, REPLICATE_MODEL, MODAL_PROXY
```

`benchmarks/.env` is gitignored -- it holds the Replicate token and any
proxy config, never commit it.

## Running

One call per invocation, on purpose: this measures cold start, so looping
here would warm the very thing being measured. Run cases by hand, one at a
time, reviewing results before moving to the next:

```bash
python benchmarks/run.py --service modal --case 1
python benchmarks/run.py --service replicate --case 1
```

Cases (defined in `payloads.py`, identical prompt/settings across services
for a given case so the comparison isolates the platform, not the input):

| Case | Description |
|---|---|
| 1 | txt2img, checkpoint only |
| 2 | txt2img, checkpoint + LoRA (HF-hosted, no rate limit) |
| 3 | txt2img, checkpoint + LoRA + OpenPose ControlNet |

Benchmark settings deliberately differ from `predict.py`'s production
defaults to cut timing noise unrelated to inference: `disable_safety_checker:
true`, `apply_watermark: false`, a fixed `seed`. Real API callers get
different (safer) defaults -- see the root README.

## Results

Each run writes to `benchmarks/results/<service>/<case>/<UTC timestamp>/`
(gitignored -- results and generated images are local-only, not committed):

- `meta.json` -- full request payload, client-measured wall clock, whatever
  server-side timing the API exposes (Replicate's `metrics`/`created_at`/
  `started_at`/`completed_at`; Modal's endpoint exposes none, cross-check the
  dashboard's Function Calls tab if needed), response shape, environment
  (hostname/platform/python), and a `cost_estimate` stub to fill in once GPU
  type + $/s rate are confirmed for that run.
- `output_N.png` -- the generated image(s), for visual sanity-checking.

## Cold start caveat

A cold-start measurement is only valid if the target was actually idle
before the call:

- **Modal**: `MIN_CONTAINERS=0` in `modal_app.py`, so any call after the
  `SCALEDOWN_WINDOW` idle period is a genuine cold start. Set to `2`
  (Modal's allowed minimum) rather than the default 60s/5min specifically so
  every benchmark call pays its own cold start with no idle-tail billing --
  see "Cost verification" below for why that matters for a fair $/call
  comparison. Not representative of production traffic, where a longer
  window trades idle billing for fewer cold starts.
- **Replicate**: cold start depends on the model's own idle/scale-to-zero
  behavior, which isn't controlled from this repo.

If a run comes back suspiciously fast, treat it as warm and note that in
place of a real cold-start number rather than reporting it as one.

## Cost verification

Neither API hands back an authoritative $ figure inline, so cost in
`meta.json` is a computed estimate (rate x billed seconds) until cross-checked
against a dashboard:

- **Modal**: no per-call cost in the API/dashboard. Two ways to check the
  real number:
  - `Usage & billing -> Usage` shows a daily total per resource (e.g. `A10G`)
    -- diff it against the pre-run total for a single call's actual cost.
  - `modal billing report --start YYYY-MM-DD --end YYYY-MM-DD --csv` gives a
    per-app-per-day total (still not per-call). Needs `pip install
    'modal[api-proxy-support]'` (pulls in `python-socks`) for the CLI's gRPC
    connection to route through `MODAL_PROXY`/`HTTPS_PROXY` on networks that
    need it -- without that package the client reads the proxy env vars but
    fails to actually use them and just times out.
  - Because `SCALEDOWN_WINDOW` billing dominates cost for infrequent calls
    (idle time is billed the same as active time), a fair Modal number
    depends entirely on what window is configured -- always record it.
- **Replicate**: `metrics.predict_time` / `metrics.total_time` on the
  prediction object are exact, but which one is *billed* depends on model
  visibility -- per replicate.com/docs/topics/billing, public models are
  billed only for `predict_time` (setup/cold-start/idle is free), private
  deployments are billed for the full `total_time`. The dashboard's
  prediction detail panel shows "Approximate cost" directly, which is the
  cheapest way to confirm which one applies.

## Adding a service

Add a `call_<service>()` function to `run.py` following the shape of
`call_modal`/`call_replicate` (same `record` schema), add it to the
`--service` choices, and it writes to
`benchmarks/results/<service>/...` alongside the rest automatically.
