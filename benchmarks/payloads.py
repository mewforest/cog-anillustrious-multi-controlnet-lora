"""Request payloads for the benchmark cases -- identical prompt/settings sent
to both services for a given case, so the comparison isolates platform
(cold start, scheduling, proxy overhead) rather than input differences.

Field names match predict.py's Predictor.predict() signature 1:1: Replicate
and Modal both run the exact same predict.py, so one dict is valid `input`
for both APIs.

Case 1 uses benchmark-friendly settings (disable_safety_checker, no
watermark) chosen to cut timing noise unrelated to inference -- see
benchmarks/README.md. Cases 2 and 3 instead reuse the prompt/negative_prompt/
settings from examples/modal/2.lora.py and the session's earlier working
OpenPose+LoRA test verbatim: the generic case-1-style prompt didn't visibly
trigger the LoRA's style, and this one (from the LoRA's own model card) is
known to. Only the seed is changed, to keep the case distinguishable from any
prior attempt with the same prompt.
"""

from __future__ import annotations

SEED = 12345
LORA_SEED = 54321  # deliberately different from SEED -- see module docstring

BASE_SETTINGS = dict(
    negative_prompt="lowres, bad anatomy, bad hands, worst quality, low quality, blurry",
    width=768,
    height=768,
    num_outputs=1,
    scheduler="K_EULER",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=SEED,
    refine="no_refiner",
    apply_watermark=False,
    disable_safety_checker=True,
)

PROMPT = (
    "masterpiece, best quality, 1girl, solo, standing, forest background, "
    "detailed background, dynamic pose"
)

# Same LyliaEngine Persona 5 style LoRA used in examples/modal/2.lora.py --
# hosted on HF (no auth, no Civitai rate limit) per the user's constraint.
HF_LORA_URL = (
    "https://huggingface.co/LyliaEngine/"
    "Persona_5_Royal_and_Strikers_2D_cutscene_art_style_Illustrious_V2/resolve/main/"
    "Persona_5_Royal_and_Strikers_2D_cutscene_art_style_Illustrious_V2.safetensors"
)

# Prompt/negative_prompt straight from the LoRA's model card, as used in
# examples/modal/2.lora.py -- the LoRA has no dedicated trigger word, so the
# style tags + character name are what actually invoke it.
LORA_PROMPT = "masterpiece, anime screencap, anime coloring, official art, Ann Takamaki"
LORA_NEGATIVE_PROMPT = (
    "bad anatomy, poorly drawn face, bad hands, morbid, deformed, "
    "disfigured, mutilated, malformed, missing body part, error, "
    "malformed hands, legs, bad feet, fused legs, broken legs, "
    "bad eyes, censored, bad body proportions, bad face, "
    "bad facial expression, gross proportions, bad abs, "
    "disappearing hands, fused hands, fused body part, fused digits, "
    "missing digit, extra digit, hand with more than 5 digits, "
    "hand with less than 5 digits, bad pecs, 3D character, "
    "photo realistic, 3D game, 3D, cropped, watermark, username, "
    "signature, not in perspective, bad artist, bad background, ugly, "
    "jpeg artifacts, squares, faded, worst quality, blurred, lowres, "
    "low quality, bad quality, plain pose, plain figure"
)

# Same pose reference as examples/modal/1.openpose.py and the session's
# earlier decisive OpenPose+LoRA test. replicate.delivery links expire --
# swap this if it 404s.
POSE_IMAGE_URL = (
    "https://replicate.delivery/pbxt/"
    "PQEr7De3SEsD5bfW4XwsF9YpPcu8NjK9oaqPql22opv0rdMH/pose8.jpg"
)


def case_1_checkpoint_only() -> dict:
    return {"prompt": PROMPT, **BASE_SETTINGS}


def case_2_checkpoint_lora() -> dict:
    # Matches examples/modal/2.lora.py's config exactly (832x1216,
    # prompt_strength 0.8, apply_watermark True, safety checker on) rather
    # than the case-1 BASE_SETTINGS -- this is the known-working
    # configuration, not the noise-reduced benchmark defaults.
    return {
        "prompt": LORA_PROMPT,
        "negative_prompt": LORA_NEGATIVE_PROMPT,
        "width": 832,
        "height": 1216,
        "sizing_strategy": "width_height",
        "num_outputs": 1,
        "scheduler": "K_EULER",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "prompt_strength": 0.8,
        "seed": LORA_SEED,
        "refine": "no_refiner",
        "apply_watermark": True,
        "lora_scale": 1.0,
        "lora_weights": HF_LORA_URL,
    }


def case_3_checkpoint_lora_openpose(pose_image_url: str = POSE_IMAGE_URL) -> dict:
    # Matches the session's earlier decisive OpenPose+LoRA test: same
    # Ann Takamaki prompt as case 2, full-body (not portrait -- see the
    # session note on pose inputs), 25 steps. 576x1024 keeps the same
    # portrait aspect as the session's original 768x1344 decisive test while
    # staying <=1024 on both dimensions -- LoRA+ControlNet at 768x1344 hit a
    # masked CUDA OOM (NVML assert) on a later run, right at A10G's memory
    # ceiling; per the user, benchmark payloads cap at 1024 on both axes.
    return {
        "prompt": LORA_PROMPT,
        "negative_prompt": LORA_NEGATIVE_PROMPT,
        "width": 576,
        "height": 1024,
        "sizing_strategy": "width_height",
        "num_outputs": 1,
        "scheduler": "K_EULER",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": LORA_SEED,
        "refine": "no_refiner",
        "apply_watermark": True,
        "lora_scale": 1.0,
        "lora_weights": HF_LORA_URL,
        "controlnet_1": "openpose_raw",
        "controlnet_1_image": pose_image_url,
        "controlnet_1_conditioning_scale": 0.75,
    }
