# Calling the Modal deployment

Three scripts against the deployed endpoint, in the order worth running them:

| Script | What it exercises |
| --- | --- |
| `3.txt2img.py` | plain txt2img, no ControlNet, no LoRA — the smoke test |
| `1.openpose.py` | OpenPose ControlNet from a URL, no LoRA |
| `2.lora.py` | a community LoRA fetched from Civitai |

```bash
pip install -r requirements.txt
cp .env.example .env   # edit if you need a proxy or a different endpoint
python 3.txt2img.py
```

Images land in `out/`. Each script sends exactly one request and never retries.

`_client.py` talks to the endpoint over plain HTTP — it does not import the
`modal` SDK, so a broken local Modal install can't get in the way.

## Environment

Config lives in `.env` (copy `.env.example` to get started) and loads
automatically — nothing to export by hand. `.env` is gitignored, since it's
the place for anything machine-specific.

| Variable | Purpose |
| --- | --- |
| `MODAL_PREDICT_URL` | endpoint URL, if it differs from the default in `_client.py` |
| `MODAL_PROXY` | proxy to dial through, e.g. `http://127.0.0.1:8080`. Unset or empty connects directly |
| `CIVITAI_API_TOKEN` | only if you point `LORA_WEIGHTS_URL` at a Civitai download link — it serves a login page without one |
| `LORA_WEIGHTS_URL` | use a different LoRA in `2.lora.py` |

A one-off override without touching `.env`:

```bash
MODAL_PROXY=off python 3.txt2img.py
```

## Responses

A single image with no ControlNet preview comes back as a raw PNG. Anything
else — several outputs, or a ControlNet preview alongside the image — comes
back as JSON with base64 `images` and `control_previews`; the client saves
both shapes.

Errors are HTTP status codes, not retries:

- **400** — bad input, e.g. an image URL that no longer resolves, or a LoRA
  the pipeline can't apply.
- **503** — the container came up but the model failed to load. The message
  carries the underlying error; it needs a fix and a redeploy, and requests
  will keep failing fast (not looping GPU containers) until then.

The first call after an idle period is the slow one: the deployment scales to
zero, so a container has to restore its snapshot before it can generate.
