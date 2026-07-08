# cog-anillustrious-multi-controlnet-lora

SDXL-architecture Cog model for [Replicate](https://replicate.com), built on the
**AnIllustrious v5** checkpoint instead of vanilla SDXL, so that the many
Illustrious/Pony-family style LoRAs on Civitai actually apply correctly.

Fork of [fofr/cog-sdxl-multi-controlnet-lora](https://github.com/fofr/cog-sdxl-multi-controlnet-lora)
(MIT) — only the base-checkpoint source changed, everything else (LoRA loading,
ControlNet, scheduler, img2img/inpaint, optional refiner) is unmodified:

- img2img
- inpainting
- custom LoRA loading (`lora_urls` / `lora_scales`)
- up to 3 simultaneous ControlNets (openpose, canny, depth, ...)
- img2img plus ControlNet
- inpainting plus ControlNet
- ControlNet conditioning strengths, start/end controls
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

## Deploying

Weights are *not* baked into the Docker image — `cog push` builds/pushes code only, and
the checkpoint is fetched from Hugging Face on first cold boot of the deployed model
(same pattern the upstream fofr cog uses for stock SDXL). See
`.github/workflows/push.yaml` (manual `workflow_dispatch`) for the CI build; it needs a
single repo secret, `REPLICATE_CLI_AUTH_TOKEN`.
