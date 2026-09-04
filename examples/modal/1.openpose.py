"""OpenPose ControlNet, no LoRA -- the same inputs as the Replicate run.

`openpose_raw` means the input is already a rendered skeleton and the detector
is skipped. Point it at an ordinary photo of a person and you want "openpose"
instead, so the detector extracts the pose first.

The response carries a ControlNet preview alongside the image, so it comes back
as JSON and the files land as openpose-0.png / openpose-control-0.png.
"""

from _client import run

if __name__ == "__main__":
    run(
        {
            "prompt": "1girl, happy, school uniform, masterpiece, full body shot",
            "negative_prompt": "lowres, bad anatomy",
            "width": 768,
            "height": 768,
            "sizing_strategy": "width_height",
            "num_outputs": 1,
            "scheduler": "K_EULER",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "prompt_strength": 0.8,
            "refine": "no_refiner",
            "apply_watermark": True,
            "lora_scale": 0.6,
            "controlnet_1": "openpose_raw",
            # Fetched server-side. replicate.delivery links do expire; swap in
            # any reachable image URL (or a base64 data URI) when it 404s.
            "controlnet_1_image": (
                "https://replicate.delivery/pbxt/"
                "PQEr7De3SEsD5bfW4XwsF9YpPcu8NjK9oaqPql22opv0rdMH/pose8.jpg"
            ),
            "controlnet_1_conditioning_scale": 0.75,
            "controlnet_1_start": 0,
            "controlnet_1_end": 1,
        },
        stem="openpose",
    )
