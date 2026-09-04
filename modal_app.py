"""
Modal deployment for this repo's AnIllustrious multi-ControlNet + LoRA SDXL
pipeline, running alongside (not instead of) the Cog/Replicate deployment.

This file is an independent entry point: it reuses predict.py, controlnet.py,
controlnet_preprocess.py, weights_manager.py, weights.py, sizing_strategy.py,
dataset_and_utils.py and no_init.py exactly as they are. It does not change
cog.yaml or how `cog push` / Replicate work.

What's different here vs. Replicate:
  - The base checkpoint, refiner, safety checker and ControlNet/preprocessor
    weights are downloaded once at *image build* time (baked into /weights),
    instead of on every cold container's first request -- that's the actual
    source of Replicate's multi-minute cold boot.
  - Per-request LoRA weights (arbitrary URLs, can't be baked in ahead of
    time) are cached on a Modal Volume shared across containers, so the same
    LoRA is only ever downloaded once across the whole deployment's lifetime.
  - The dependency stack (torch/diffusers/transformers/...) is newer than the
    one pinned in cog.yaml for Replicate.

Local prerequisites (on your machine, not in the container):
    pip install modal fastapi pydantic
    modal setup   # one-time auth

Deploy:
    modal deploy modal_app.py

Test:
    curl -X POST <endpoint-url printed by `modal deploy`> \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "a fox in a snowy forest, anime style"}' \
        --output out.png
"""

import base64
import os
import re
import tempfile
from typing import Optional

import modal
from pydantic import BaseModel

app = modal.App("anillustrious-multi-controlnet-lora")

# Everything baked in at image-build time lives under here. predict.py,
# controlnet.py and controlnet_preprocess.py all reference their weight
# caches with paths relative to the process cwd (e.g. "./sdxl-cache"), so
# making this the container's working directory is enough to make those
# unmodified constants resolve here -- no need to touch that code.
CACHE_ROOT = "/weights"

# Per-request LoRA downloads can't be baked in ahead of time (arbitrary URLs
# supplied by the caller), so they get their own persistent, shared cache.
LORA_CACHE_DIR = "/cache/lora-cache"
LORA_VOLUME_NAME = "anillustrious-lora-cache"

GPU_TYPE = "A10G"  # bump to "L40S" / "A100" if generation speed matters more than cost
MIN_CONTAINERS = 0  # set to 1+ to keep a container permanently warm (no cold starts, but billed idle time)
SCALEDOWN_WINDOW = 2  # seconds a warm container is kept around after its last request (2s = Modal's allowed minimum; every request pays cold start, but no idle-tail billing -- set for a fair per-call cost benchmark against Replicate's public-model pricing, not for production traffic)

# GPU memory snapshotting is an experimental Modal feature that restores
# already-initialized GPU state instead of re-loading+moving weights on cold
# start (see https://modal.com/docs/guide/memory-snapshots). Flip to False if
# it misbehaves for this GPU/driver combination -- CPU-only memory snapshot
# (enable_memory_snapshot below) still applies either way.
ENABLE_GPU_SNAPSHOT = True


def _bake_base_and_refiner_weights():
    """Runs once at image-build time: downloads the base checkpoint, refiner
    and safety checker into the image so cold containers never fetch them
    over the network."""
    os.chdir(CACHE_ROOT)
    from predict import (
        ILLUSTRIOUS_REPO_ID,
        REFINER_MODEL_CACHE,
        REFINER_URL,
        SAFETY_CACHE,
        SAFETY_URL,
        SDXL_MODEL_CACHE,
    )
    from weights_downloader import WeightsDownloader

    WeightsDownloader.download_hf_snapshot_if_not_exists(ILLUSTRIOUS_REPO_ID, SDXL_MODEL_CACHE)
    WeightsDownloader.download_if_not_exists(REFINER_URL, REFINER_MODEL_CACHE)
    WeightsDownloader.download_if_not_exists(SAFETY_URL, SAFETY_CACHE)


def _bake_controlnet_weights():
    """Runs once at image-build time: pre-warms the same ControlNet and
    preprocessor caches controlnet.py / controlnet_preprocess.py read from at
    request time, so picking any of the supported ControlNet types never
    triggers a download on a cold container."""
    os.chdir(CACHE_ROOT)
    from controlnet import CONTROLNET_MODEL_CACHE, CONTROLNET_URL
    from controlnet_preprocess import (
        CONTROLNET_PREPROCESSOR_MODEL_CACHE,
        CONTROLNET_PREPROCESSOR_URL,
    )
    from weights_downloader import WeightsDownloader

    WeightsDownloader.download_if_not_exists(CONTROLNET_URL, CONTROLNET_MODEL_CACHE)
    WeightsDownloader.download_if_not_exists(
        CONTROLNET_PREPROCESSOR_URL, CONTROLNET_PREPROCESSOR_MODEL_CACHE
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",  # libgthread-2.0.so.0 — required by opencv-python
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "curl",
    )
    .run_commands(
        "curl -o /usr/local/bin/pget -L "
        "https://github.com/replicate/pget/releases/download/v0.0.6/pget "
        "&& chmod +x /usr/local/bin/pget"
    )
    .pip_install(
        "cog",  # only needed so predict.py's `from cog import ...` resolves unmodified
        "torch==2.5.1",
        "torchvision==0.20.1",
        "diffusers==0.31.0",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        # diffusers 0.31 routes load_lora_weights() exclusively through the
        # PEFT backend and raises "PEFT backend is required for this method"
        # without it. (Replicate's diffusers 0.21.4 predates that and uses its
        # own loader, which is why cog.yaml doesn't list peft.) The backend
        # switches on at peft >= 0.6 + transformers >= 4.34.
        "peft==0.13.2",
        "invisible-watermark==0.2.0",
        "numpy==1.26.4",
        "opencv-python-headless>=4.1.0.25",
        "controlnet-aux==0.0.9",
        "mediapipe==0.10.14",
        "huggingface_hub==0.26.2",
        "requests==2.32.3",
        "safetensors",
        "fastapi[standard]",
    )
    .env({"HF_HUB_DOWNLOAD_TIMEOUT": "120"})
    .workdir(CACHE_ROOT)
    .add_local_python_source(
        "predict",
        "controlnet",
        "controlnet_preprocess",
        "weights_downloader",
        "weights_manager",
        "weights",
        "sizing_strategy",
        "dataset_and_utils",
        "no_init",
        copy=True,  # must be baked into the image layer so run_function() below can import them
    )
    .add_local_dir("feature-extractor", remote_path=f"{CACHE_ROOT}/feature-extractor", copy=True)
    .run_function(_bake_base_and_refiner_weights)
    .run_function(_bake_controlnet_weights)
)

lora_volume = modal.Volume.from_name(LORA_VOLUME_NAME, create_if_missing=True)


class PredictRequest(BaseModel):
    """Mirrors predict.py's Predictor.predict() inputs. Anything left unset
    (None) falls back to that function's own cog.Input(...) default."""

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    image: Optional[str] = None  # URL or base64 (data URI or raw), for img2img/inpaint
    mask: Optional[str] = None  # URL or base64, for inpaint
    width: Optional[int] = None
    height: Optional[int] = None
    sizing_strategy: Optional[str] = None
    num_outputs: Optional[int] = None
    scheduler: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    prompt_strength: Optional[float] = None
    seed: Optional[int] = None
    refine: Optional[str] = None
    refine_steps: Optional[int] = None
    apply_watermark: Optional[bool] = None
    lora_scale: Optional[float] = None
    lora_weights: Optional[str] = None
    disable_safety_checker: Optional[bool] = None
    controlnet_1: Optional[str] = None
    controlnet_1_image: Optional[str] = None
    controlnet_1_conditioning_scale: Optional[float] = None
    controlnet_1_start: Optional[float] = None
    controlnet_1_end: Optional[float] = None
    controlnet_2: Optional[str] = None
    controlnet_2_image: Optional[str] = None
    controlnet_2_conditioning_scale: Optional[float] = None
    controlnet_2_start: Optional[float] = None
    controlnet_2_end: Optional[float] = None
    controlnet_3: Optional[str] = None
    controlnet_3_image: Optional[str] = None
    controlnet_3_conditioning_scale: Optional[float] = None
    controlnet_3_start: Optional[float] = None
    controlnet_3_end: Optional[float] = None


IMAGE_INPUT_FIELDS = (
    "image",
    "mask",
    "controlnet_1_image",
    "controlnet_2_image",
    "controlnet_3_image",
)


def _materialize_image_input(value: str) -> str:
    """predict.py's image inputs are cog.Path (file paths); the JSON API
    instead accepts a URL or a base64-encoded image, so fetch/decode it to a
    temp file and hand back that path."""
    if value.startswith("http://") or value.startswith("https://"):
        import requests

        resp = requests.get(value, timeout=(10, 60))
        resp.raise_for_status()
        data = resp.content
    else:
        data = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", value))

    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={LORA_CACHE_DIR: lora_volume},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=600,
    # One request means one attempt: a failed prediction comes back to the
    # caller as an error, it is never re-run. (This only covers exceptions
    # raised *inside* predict() -- a container that crashes is rescheduled by
    # Modal regardless of this setting, which is why load() below refuses to
    # raise.)
    retries=0,
    enable_memory_snapshot=True,
    experimental_options=({"enable_gpu_snapshot": True} if ENABLE_GPU_SNAPSHOT else {}),
)
class IllustriousXLModel:
    @modal.enter(snap=True)
    def load(self):
        """Never raises.

        Anything thrown out of a lifecycle method kills the container, and
        Modal responds by rescheduling the container *and the work it was
        assigned* -- for a deployed App, indefinitely
        (https://modal.com/docs/guide/retries). A setup bug therefore turns a
        single request into an unbounded start/crash/restart loop that keeps
        booting GPU containers, and cancelling the function call does not stop
        the attempts that are already queued -- it has to be killed by hand
        from the dashboard while billing runs.

        So the failure is caught and remembered here instead, and predict()
        reports it as a plain HTTP error: one request, one container, one
        answer. The recorded error survives into the memory snapshot, so once
        setup is broken every container answers 503 immediately (cheap) until
        the cause is fixed and the app is redeployed.
        """
        self._setup_error = None
        try:
            self._setup()
        except Exception as e:
            import traceback

            traceback.print_exc()
            self._setup_error = f"{type(e).__name__}: {e}"

    def _setup(self):
        os.chdir(CACHE_ROOT)

        from predict import Predictor

        self.predictor = Predictor()
        self.predictor.setup()

        # Calling a Cog Predictor directly (bypassing cog's own HTTP layer)
        # means an unset kwarg falls back to the raw cog.Input(...)
        # descriptor object, not its intended default value -- so every
        # predict() call below must pass every argument explicitly. Derive
        # those defaults from the live signature instead of hardcoding a
        # second copy that could drift out of sync with predict.py.
        import inspect

        from cog.input import FieldInfo

        self._predict_defaults = {
            name: (
                param.default.default
                if isinstance(param.default, FieldInfo)
                else param.default
            )
            for name, param in inspect.signature(Predictor.predict).parameters.items()
            if name != "self"
        }

    @modal.enter(snap=False)
    def attach_lora_cache(self):
        """Runs after a snapshot restore, once per container.

        setup() built the default plain-local-disk LoRA cache
        (weights_manager.py); this swaps in the Volume-backed one so LoRA
        downloads survive container restarts and are shared across containers.
        It deliberately lives outside the snapshotted method: the modal.Volume
        handle and the directory listing the cache reads when constructed are
        live per-container state, not something to freeze into an image-wide
        snapshot.
        """
        if self._setup_error is not None:
            return

        from weights import WeightsDownloadCache

        self.predictor.weights_manager.weights_cache = WeightsDownloadCache(
            base_dir=LORA_CACHE_DIR, volume=lora_volume
        )

    # requires_proxy_auth: without it this URL is an open GPU — anyone who
    # learns it can spend the workspace's money. The URL is not a secret and
    # is committed in this repo's clients; the Modal-Key / Modal-Secret proxy
    # token pair is. Create one in the Modal dashboard under
    # Settings -> Proxy Auth Tokens and send both headers on every call.
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def predict(self, request: PredictRequest):
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        if self._setup_error is not None:
            # See load(): a setup failure is reported, not crashed on.
            raise HTTPException(
                status_code=503,
                detail=(
                    "model setup failed on this container, so no prediction can "
                    f"run: {self._setup_error}. Fix the cause and redeploy."
                ),
            )

        overrides = request.model_dump(exclude_none=True)
        for field in IMAGE_INPUT_FIELDS:
            if field in overrides:
                try:
                    overrides[field] = _materialize_image_input(overrides[field])
                except Exception as e:
                    # Bad/expired image URL is the caller's problem, not a
                    # server error -- say so rather than returning a 500.
                    raise HTTPException(
                        status_code=400, detail=f"could not read {field}: {e}"
                    ) from e

        kwargs = {**self._predict_defaults, **overrides}

        try:
            output_paths = self.predictor.predict(**kwargs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        control_previews = [
            p for p in output_paths if os.path.basename(str(p)).startswith("control-")
        ]
        images = [p for p in output_paths if p not in control_previews]

        # Common case (single output, no controlnet preview images): hand
        # back the raw PNG directly instead of a JSON wrapper.
        if len(images) == 1 and not control_previews:
            with open(images[0], "rb") as f:
                return Response(content=f.read(), media_type="image/png")

        def _b64(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()

        return JSONResponse(
            {
                "images": [_b64(p) for p in images],
                "control_previews": [_b64(p) for p in control_previews],
            }
        )
