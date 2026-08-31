"""Persona 5 Royal & Strikers cutscene style LoRA, on top of plain txt2img.

Two things to know about how this model takes a LoRA:

1. The `<lora:Name:1>` tag people paste into A1111 prompts means nothing here --
   it is not parsed, it would just be noise in the prompt. The file goes in
   `lora_weights` as a URL, and the `:1` weight goes in `lora_scale`.
2. This particular LoRA has no trigger word (its model card says "None"): the
   style tags and the character name in the prompt are all it needs.

The default URL is the HuggingFace mirror, which serves the file to anyone.
Civitai (model 1247653, version 1505235) hosts the same weights but answers
unauthenticated clients with a login page, so it needs CIVITAI_API_TOKEN --
set LORA_WEIGHTS_URL to that (or to any other .safetensors) to override.

The first request for a given URL downloads it onto the deployment's shared
Volume; later requests reuse it.
"""

import os

from _client import run

HF_LORA_URL = (
    "https://huggingface.co/LyliaEngine/"
    "Persona_5_Royal_and_Strikers_2D_cutscene_art_style_Illustrious_V2/resolve/main/"
    "Persona_5_Royal_and_Strikers_2D_cutscene_art_style_Illustrious_V2.safetensors"
)


def lora_weights_url() -> str:
    return os.environ.get("LORA_WEIGHTS_URL", HF_LORA_URL)


if __name__ == "__main__":
    run(
        {
            # Prompt and negative prompt as given on the LoRA's model card,
            # minus the <lora:...:1> tag -- see the note at the top.
            "prompt": (
                "masterpiece, anime screencap, anime coloring, official art, "
                "Ann Takamaki"
            ),
            "negative_prompt": (
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
            "lora_scale": 1.0,  # the ":1" from the model card's <lora:...:1>
            "lora_weights": lora_weights_url(),
        },
        stem="lora",
    )
