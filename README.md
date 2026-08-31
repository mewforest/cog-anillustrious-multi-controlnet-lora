# cog-anillustrious-multi-controlnet-lora

SDXL-architecture Cog model for [Replicate](https://replicate.com), built on the
**AnIllustrious v5** checkpoint instead of vanilla SDXL, so that the many
Illustrious/Pony-family style LoRAs on Civitai actually apply correctly.

Fork of [fofr/cog-sdxl-multi-controlnet-lora](https://github.com/fofr/cog-sdxl-multi-controlnet-lora)
(MIT) — only the base-checkpoint source changed, everything else (LoRA loading,
ControlNet, scheduler, img2img/inpaint, optional refiner) is unmodified:

- img2img
- inpainting
- LoRA loading (`lora_weights` / `lora_scale`) — see "LoRA support" below
- up to 3 simultaneous ControlNets (openpose, canny, depth, ...)
- img2img plus ControlNet
- inpainting plus ControlNet
- ControlNet conditioning strengths, start/end controls

### ControlNet types: `openpose` vs `openpose_raw`

Both use the same SDXL weights (`thibaud/controlnet-openpose-sdxl-1.0`):

| Type | Input image | Preprocessing |
|------|-------------|---------------|
| `openpose` | Photograph of a person | Runs `OpenposeDetector` to extract a skeleton |
| `openpose_raw` | Already-rendered OpenPose skeleton (white limbs on black) | Pass-through — no detector |

Use `openpose_raw` when the caller already drew/exported keypoints (e.g. Miracle Studio's `poses.control_image_b64`). Feeding a finished skeleton to plain `openpose` makes the detector find no person and silently blank the conditioning image.
- optional SDXL refiner pass
- image resizing based on width/height, input image, or a control image
- disable safety checker via API

## Base checkpoint

[AnIllustrious v5](https://civitai.com/models/1121885/anillustrious?modelVersionId=1652528)
by [Jedas](https://civitai.com/user/Jedas), an Illustrious/NoobAI-family anime SDXL
finetune. Loaded from the public diffusers-format mirror
[`John6666/anillustrious-v5vae-sdxl`](https://huggingface.co/John6666/anillustrious-v5vae-sdxl)
on the Hugging Face Hub, downloaded at `setup()` time (no auth needed — the mirror is
public/ungated).

Civitai's own `/api/download/models/...` route no longer accepts API-key auth from a
plain HTTP client (it redirects to `/login` regardless of a valid token, even for
non-gated files), which is why this fork pulls from the HF mirror rather than Civitai
directly.

License for the checkpoint itself: [`faipl-1.0-sd`](https://freedevproject.org/faipl-1.0-sd/)
(Fair AI Public License 1.0-SD).

## LoRA support

`lora_weights` accepts a URL to either:

- a plain **community LoRA/LoCon `.safetensors` file** (what ~all Civitai style/character
  LoRAs are — standard Kohya-format keys), loaded via diffusers' native
  `pipe.load_lora_weights()`, or
- a **Replicate-trainer bundle** (the tar produced by Replicate's own SDXL
  dreambooth+LoRA fine-tuning API: `unet.safetensors`/`lora.safetensors` +
  `special_params.json` + `embeddings.pti`), loaded via the original fofr hand-rolled
  path — kept as a fallback, not the primary case for this anime-focused fork.

The format is auto-detected from the downloaded file's content (not the URL), so no
extra input flag is needed. True LyCORIS algorithms (LoHa/LoKr) are not supported —
loading one raises a clear error instead of silently applying nothing.

For Civitai LoRA links, append `?token=<your Civitai API key>` (from
[civitai.com/user/account](https://civitai.com/user/account) → API Keys) to the download
URL passed as `lora_weights`.

## Deploying

### Replicate (Cog)

Weights are *not* baked into the Docker image — `cog push` builds/pushes code only, and
the checkpoint is fetched from Hugging Face on first cold boot of the deployed model
(same pattern the upstream fofr cog uses for stock SDXL). See
`.github/workflows/push.yaml` (manual `workflow_dispatch`) for the CI build; it needs a
single repo secret, `REPLICATE_CLI_AUTH_TOKEN`.

### Modal

`modal_app.py` is an independent deployment of the same model on
[Modal](https://modal.com), for faster cold starts than Replicate's public-model queue.
It reuses all the business logic in this repo unmodified (`predict.py`, `controlnet.py`,
`weights_manager.py`, ...) — this is a second entry point, not a replacement for the Cog
deployment above; `cog.yaml`/`predict.py` keep working exactly as before.

Key differences from the Replicate deployment:
- The base checkpoint, refiner, safety checker and ControlNet/preprocessor weights are
  downloaded once at *image build* time and baked into the image (`/weights`), instead of
  on every cold container — that's what actually causes Replicate's multi-minute cold
  boot here.
- Per-request `lora_weights` (arbitrary URLs) still can't be baked in ahead of time, so
  they're cached on a Modal Volume shared across containers instead of a purely local,
  per-container disk cache.
- Dependency stack (torch/diffusers/transformers) is newer than the version pinned in
  `cog.yaml` for Replicate.

Deploy manually:

```bash
pip install modal fastapi pydantic
modal setup            # one-time authentication
modal deploy modal_app.py
```

Or via CI: `.github/workflows/modal-deploy.yaml` (manual `workflow_dispatch`), which
needs two repo secrets, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from
`modal token new` or your Modal workspace settings).

The deployed endpoint takes a JSON POST body mirroring `predict()`'s inputs (see
`PredictRequest` in `modal_app.py`) — `image`/`mask`/`controlnet_N_image` accept either a
URL or a base64-encoded image. It returns a raw PNG for the single-image case, or a JSON
body with base64-encoded `images`/`control_previews` otherwise:

```bash
curl -X POST <endpoint-url printed by `modal deploy`> \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "a fox in a snowy forest, anime style"}' \
    --output out.png
```

`examples/modal/` has ready-to-run Python versions of that call — plain txt2img,
ControlNet and LoRA — including proxy handling and saving both response shapes.

A prediction is attempted exactly once. Errors come back as HTTP status codes: 400
for bad input (an unreachable image URL, an unloadable LoRA), 503 if the model
itself failed to load, which needs a fix and a redeploy.
