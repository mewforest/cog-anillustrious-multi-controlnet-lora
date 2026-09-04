"""Shared config/IO helpers for the Modal-vs-Replicate (and future services)
benchmark harness. Loads config from benchmarks/.env (gitignored) -- see
.env.example for the keys. Never print or log secret values.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"

MODAL_DEFAULT_URL = (
    "https://mew-forest--anillustrious-multi-controlnet-lora-illustri-82bbe6.modal.run"
)


def modal_predict_url() -> str:
    return os.environ.get("MODAL_PREDICT_URL", MODAL_DEFAULT_URL)


def modal_auth_headers() -> dict[str, str]:
    """Proxy-auth headers for the Modal endpoint.

    The endpoint is declared `requires_proxy_auth=True`, so Modal rejects an
    unauthenticated request before any GPU container starts. The endpoint URL
    is not a secret (it is hardcoded above and committed); this token pair is.
    Create one at Modal dashboard -> Settings -> Proxy Auth Tokens.
    """
    key = os.environ.get("MODAL_KEY", "")
    secret = os.environ.get("MODAL_SECRET", "")
    if not key or not secret:
        raise SystemExit(
            "MODAL_KEY / MODAL_SECRET are not set. Create a proxy auth token in "
            "the Modal dashboard (Settings -> Proxy Auth Tokens) and put both "
            "values in benchmarks/.env -- see .env.example."
        )
    return {"Modal-Key": key, "Modal-Secret": secret}


def modal_proxy() -> str | None:
    raw = os.environ.get("MODAL_PROXY", "")
    if raw.strip().lower() in ("", "0", "off", "false", "none", "no"):
        return None
    return raw


def replicate_api_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise SystemExit(
            "REPLICATE_API_TOKEN is not set. Copy benchmarks/.env.example to "
            "benchmarks/.env and fill it in."
        )
    return token


def replicate_model() -> str:
    model = os.environ.get("REPLICATE_MODEL", "")
    if not model:
        raise SystemExit(
            "REPLICATE_MODEL is not set (owner/name of the pushed Cog model). "
            "Set it in benchmarks/.env."
        )
    return model


def civitai_download_url(version_id: int) -> str:
    """Build an authenticated Civitai LoRA download URL.

    Same shape as the Miracle001 project's build_civitai_lora_download_url:
    the model-version id resolves the right file via type/format params
    rather than a specific fileId, plus a token so unauthenticated redirects
    (Civitai sends anonymous /api/download requests to a login page) don't
    break the fetch. Never print/log the returned URL -- it carries the
    token in cleartext.
    """
    token = os.environ.get("CIVITAI_TOKEN", "")
    if not token:
        raise SystemExit(
            "CIVITAI_TOKEN is not set. Copy the CIVITAI_TOKEN line from "
            "benchmarks/.env.example into benchmarks/.env and fill it in."
        )
    return f"https://civitai.com/api/download/models/{version_id}?type=Model&format=SafeTensor&token={token}"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(service: str, case: str) -> Path:
    d = RESULTS_DIR / service / case / now_stamp()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(d: Path, record: dict, images: list[tuple[str, bytes]]) -> None:
    """images: (kind, bytes) pairs, kind e.g. "output" or "control_preview"."""
    counts: dict[str, int] = {}
    for kind, img in images:
        i = counts.get(kind, 0)
        counts[kind] = i + 1
        (d / f"{kind}_{i}.png").write_bytes(img)
    (d / "meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"saved {d}")


def client_env() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def get_with_retry(
    client: "object", url: str, *, headers: dict | None = None, attempts: int = 6
) -> "object":
    """GET with retries for transient network errors (ReadTimeout,
    RemoteProtocolError, ConnectError). Only ever used for idempotent reads
    (status polling, downloading output) -- never for the POST that creates
    a prediction/triggers a run, so a retry can't double-bill anything.
    """
    import time as _time

    import httpx as _httpx

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.get(url, headers=headers) if headers else client.get(url)
            resp.raise_for_status()
            return resp
        except _httpx.HTTPError as e:
            last_exc = e
            print(f"  retry {attempt + 1}/{attempts} for GET {url}: {type(e).__name__}: {e}")
            _time.sleep(min(2 * (attempt + 1), 10))
    raise SystemExit(f"GET {url} failed after {attempts} attempts: {last_exc}")


def empty_cost_estimate(note: str) -> dict:
    return {
        "gpu_type": None,
        "rate_usd_per_second": None,
        "estimated_usd": None,
        "note": note,
    }
