"""Plain txt2img -- no ControlNet, no LoRA. The smoke test to run first."""

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
            "refine": "no_refiner",
            "apply_watermark": True,
        },
        stem="txt2img",
    )
