"""Run one benchmark case against Modal or Replicate and record full metadata.

Usage:
    python benchmarks/run.py --service modal --case 1
    python benchmarks/run.py --service replicate --case 1

Deliberately one call per invocation: this is a cold-start benchmark, so
looping or retrying here would warm the very thing being measured. If a call
fails, it's reported and left alone -- rerun by hand once diagnosed.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import certifi
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import payloads

CASES = {
    1: ("case1_checkpoint_only", lambda: payloads.case_1_checkpoint_only()),
    2: ("case2_checkpoint_lora", lambda: payloads.case_2_checkpoint_lora()),
    3: ("case3_checkpoint_lora_openpose", lambda: payloads.case_3_checkpoint_lora_openpose()),
}

HTTP_TIMEOUT_S = 660.0


def call_modal(payload: dict) -> tuple[dict, list[bytes]]:
    url = common.modal_predict_url()
    proxy = common.modal_proxy()

    t0 = time.time()
    t0_iso = datetime.now(timezone.utc).isoformat()
    with httpx.Client(
        proxy=proxy,
        timeout=HTTP_TIMEOUT_S,
        verify=certifi.where(),
        trust_env=False,
        follow_redirects=True,  # a long-running call can come back as a 303 (proxy/gateway
        # timeout on the underlying connection, not an app-level redirect) -- httpx
        # does not follow redirects by default, and swallowing that silently meant
        # an earlier run saved an empty 0-byte PNG instead of the real image.
    ) as client:
        resp = client.post(url, json=payload)
    t1 = time.time()
    t1_iso = datetime.now(timezone.utc).isoformat()

    if resp.status_code >= 400:
        raise SystemExit(f"Modal HTTP {resp.status_code}: {resp.text}")
    if resp.status_code != 200:
        print(f"  note: unexpected status {resp.status_code} after redirects, history={[r.status_code for r in resp.history]}")

    images: list[tuple[str, bytes]] = []
    if "application/json" in resp.headers.get("content-type", ""):
        data = resp.json()
        for b64 in data.get("images") or []:
            images.append(("output", base64.b64decode(b64)))
        for b64 in data.get("control_previews") or []:
            images.append(("control_preview", base64.b64decode(b64)))
    else:
        images.append(("output", resp.content))

    record = {
        "service": "modal",
        "request": {"url": url, "proxy_used": bool(proxy), "payload": payload},
        "timing": {
            "client_request_sent_at": t0_iso,
            "client_response_received_at": t1_iso,
            "client_wall_clock_seconds": round(t1 - t0, 3),
            "server_metrics": {
                "note": (
                    "Modal's web endpoint returns no per-call timing breakdown. "
                    "Cross-check cold-start vs execution split in the app's "
                    "dashboard Function Calls tab if needed."
                )
            },
        },
        "response": {
            "status_code": resp.status_code,
            "num_images": len(images),
            "content_type": resp.headers.get("content-type"),
        },
        "cost_estimate": common.empty_cost_estimate(
            "fill in via benchmarks/costs.py once GPU type + $/s rate are confirmed "
            "(A10G ~= $0.000306/s per prior session research -- verify at modal.com/pricing)"
        ),
    }
    return record, images


def call_replicate(payload: dict) -> tuple[dict, list[bytes]]:
    token = common.replicate_api_token()
    model = common.replicate_model()
    proxy = common.modal_proxy()  # same network path needed for replicate.delivery too
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # The model-scoped POST /v1/models/{owner}/{name}/predictions endpoint
    # 404s for this (non-"official") pushed Cog model, so resolve the latest
    # version id first and use the classic version-based create endpoint.
    with httpx.Client(
        proxy=proxy, timeout=30.0, verify=certifi.where(), trust_env=False
    ) as client:
        model_resp = common.get_with_retry(
            client, f"https://api.replicate.com/v1/models/{model}", headers=headers
        )
        version_id = model_resp.json()["latest_version"]["id"]

    create_url = "https://api.replicate.com/v1/predictions"

    t0 = time.time()
    t0_iso = datetime.now(timezone.utc).isoformat()
    with httpx.Client(
        proxy=proxy, timeout=HTTP_TIMEOUT_S, verify=certifi.where(), trust_env=False
    ) as client:
        # Not retried -- a retry here could create a second billed prediction.
        create = client.post(
            create_url,
            headers=headers,
            json={"version": version_id, "input": payload},
        )
        if create.status_code >= 400:
            raise SystemExit(f"Replicate HTTP {create.status_code}: {create.text}")
        pred = create.json()
        pred_id = pred["id"]
        print(f"  created prediction {pred_id}")

        # No `Prefer: wait` -- polling from a cold model IS the measurement.
        # Polling itself is retried freely: it's a read, it can't double-bill.
        while pred["status"] not in ("succeeded", "failed", "canceled"):
            time.sleep(2)
            poll = common.get_with_retry(
                client,
                f"https://api.replicate.com/v1/predictions/{pred_id}",
                headers=headers,
            )
            pred = poll.json()
    t1 = time.time()
    t1_iso = datetime.now(timezone.utc).isoformat()

    if pred["status"] != "succeeded":
        raise SystemExit(
            f"Replicate prediction {pred_id} ended as {pred['status']}: {pred.get('error')}"
        )

    output = pred["output"]
    urls = output if isinstance(output, list) else [output]
    images: list[tuple[str, bytes]] = []
    with httpx.Client(
        proxy=proxy, timeout=60.0, verify=certifi.where(), trust_env=False
    ) as client:
        for u in urls:
            r = common.get_with_retry(client, u)
            images.append(("output", r.content))

    record = {
        "service": "replicate",
        "request": {
            "url": create_url,
            "model": model,
            "version": version_id,
            "payload": payload,
        },
        "timing": {
            "client_request_sent_at": t0_iso,
            "client_response_received_at": t1_iso,
            "client_wall_clock_seconds": round(t1 - t0, 3),
            "server_metrics": {
                "created_at": pred.get("created_at"),
                "started_at": pred.get("started_at"),
                "completed_at": pred.get("completed_at"),
                "metrics": pred.get("metrics"),
            },
        },
        "response": {
            "status_code": 200,
            "num_images": len(images),
            "prediction_id": pred_id,
        },
        "cost_estimate": common.empty_cost_estimate(
            "fill in via benchmarks/costs.py once the model's hardware tier + "
            "$/s rate are confirmed (see the model page's Hardware badge on Replicate)"
        ),
    }
    return record, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=["modal", "replicate"], required=True)
    parser.add_argument("--case", type=int, choices=sorted(CASES), required=True)
    args = parser.parse_args()

    case_slug, payload_fn = CASES[args.case]
    payload = payload_fn()

    print(f"=== {args.service} / {case_slug} ===")
    print(f"payload: {payload}")

    caller = call_modal if args.service == "modal" else call_replicate
    record, images = caller(payload)
    record["case"] = case_slug
    record["environment"] = common.client_env()

    d = common.run_dir(args.service, case_slug)
    common.save_result(d, record, images)

    print(f"wall clock: {record['timing']['client_wall_clock_seconds']}s")
    print(f"server metrics: {record['timing']['server_metrics']}")


if __name__ == "__main__":
    main()
