"""Persona 5 Illustrious LoRA -- the Replicate prompt plus a real weights URL.

The original Replicate run only carried `<lora:...:1>` inside the prompt. This
model does not parse A1111 tag syntax: that text was just text, and no LoRA was
ever applied. Loading one takes a .safetensors URL in `lora_weights`.

Version 1505235 of civitai.com/models/1247653 is the file the tag named:
Persona_5_Royal_and_Strikers_2D_cutscene_art_style_Illustrious_V2.safetensors.

Civitai serves a login page instead of the file to unauthenticated clients, so
set CIVITAI_API_TOKEN (civitai.com/user/account -> API Keys), or point
LORA_WEIGHTS_URL at any other reachable .safetensors.

The first request for a given LoRA downloads it onto the deployment's shared
Volume; later requests for the same URL reuse it.
"""

import os

from _client import run

LORA_VERSION_ID = "1505235"


def lora_weights_url() -> str:
    if url := os.environ.get("LORA_WEIGHTS_URL"):
        return url
    token = os.environ.get("CIVITAI_API_TOKEN")
    if not token:
        raise SystemExit(
            "Set CIVITAI_API_TOKEN or LORA_WEIGHTS_URL. "
            "Token: civitai.com/user/account -> API Keys."
        )
    return (
        f"https://civitai.com/api/download/models/{LORA_VERSION_ID}"
        f"?type=Model&format=SafeTensor&token={token}"
    )


if __name__ == "__main__":
    run(
        {
            "prompt": (
                "anime coloring, anime screencap, from above, "
                "1girl, solo, from side, head tilt, 18yo, laboratory, "
                "aegis_(persona), persona, blonde_hair, blue_eyes, robot, "
                "white outfit, neutral expression, nail polish, frilled choker, "
                "bow, bare shoulders, black dress, thighhighs, cross legs, "
                "one hand on knee (looking at viewer), dynamic pose, "
                "(depth of field), cool blue theme, yellow highlight, "
                "masterpiece, best quality, amazing quality"
            ),
            "negative_prompt": (
                "blurry, lowres, bad anatomy, extra fingers, watermark, "
                "text, multiple people"
            ),
            "width": 832,
            "height": 1216,
            "sizing_strategy": "width_height",
            "num_outputs": 1,
            "scheduler": "K_EULER",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "prompt_strength": 0.8,
            "refine": "no_refiner",
            "apply_watermark": True,
            "lora_scale": 0.6,
            "lora_weights": lora_weights_url(),
        },
        stem="lora",
    )
