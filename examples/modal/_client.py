"""POST a PredictRequest to the deployed Modal web endpoint.

Deliberately does not import the `modal` SDK: it isn't needed to call a public
web endpoint, and on at least one machine `import modal` blows up because
another project's venv leaks an ancient typing_extensions onto sys.path.

Nothing here retries. One run of a script sends exactly one request, so a
failure costs one container, not a stream of them.

Config lives in `.env` (see `.env.example`), loaded automatically:
    MODAL_PREDICT_URL   endpoint URL (default below)
    MODAL_PROXY         proxy URL to dial through, e.g. http://127.0.0.1:8080;
                        unset or empty connects directly
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import certifi
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Modal derives the hostname from workspace + app + class + method, which here
# comes to 74 characters -- longer than the 63 a DNS label allows -- so it
# truncates to the first 56 and appends a hash: this is the URL the deployment
# actually serves on. Check the `modal deploy` output or the app's dashboard
# page if it ever changes, and set MODAL_PREDICT_URL rather than editing this.
DEFAULT_URL = (
    "https://mew-forest--anillustrious-multi-controlnet-lora-illustri-82bbe6.modal.run"
)
OUT_DIR = Path(__file__).resolve().parent / "out"

# The endpoint's own limit is 600s; wait a little longer so a server-side
# timeout comes back as the server's error message instead of being masked by
# the client giving up first. A cold container has to restore its snapshot
# before it can generate, so the first call after an idle period is the slow one.
TIMEOUT_S = 660.0


def predict_url() -> str:
    return os.environ.get("MODAL_PREDICT_URL", DEFAULT_URL)


def proxy_url() -> str | None:
    raw = os.environ.get("MODAL_PROXY", "")
    if raw.strip().lower() in ("", "0", "off", "false", "none", "no"):
        return None
    return raw


def run(payload: dict, stem: str) -> list[Path]:
    url = predict_url()
    proxy = proxy_url()
    print(f"POST  {url}")
    print(f"proxy {proxy or 'off'}")

    start = time.time()
    try:
        with httpx.Client(
            proxy=proxy,
            timeout=TIMEOUT_S,
            verify=certifi.where(),
            trust_env=False,  # ignore any HTTPS_PROXY leftover in the shell
            follow_redirects=True,  # a long-running call can come back as a 303
            # (proxy/gateway timeout on the underlying connection); httpx doesn't
            # follow redirects by default, which silently turns that into an
            # empty response instead of the real image.
        ) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError as e:
        hint = (
            f"could not reach the proxy at {proxy} -- is it running? "
            "Set MODAL_PROXY=off in .env to connect directly."
            if proxy
            else "could not reach the endpoint directly; if you're behind a "
            "VPN/firewall, set MODAL_PROXY in .env (see .env.example)."
        )
        raise SystemExit(f"{e}\n{hint}") from e

    elapsed = time.time() - start
    print(f"HTTP {resp.status_code} in {elapsed:.1f}s")

    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = json.dumps(resp.json(), indent=2, ensure_ascii=False)
        except ValueError:
            pass
        raise SystemExit(f"HTTP {resp.status_code}: {detail}")

    OUT_DIR.mkdir(exist_ok=True)
    written: list[Path] = []

    # The endpoint returns a raw PNG for the common single-image case and a
    # JSON envelope when there is more than one image or a ControlNet preview.
    if "application/json" in resp.headers.get("content-type", ""):
        data = resp.json()
        for i, b64 in enumerate(data.get("images") or []):
            path = OUT_DIR / f"{stem}-{i}.png"
            path.write_bytes(base64.b64decode(b64))
            written.append(path)
        for i, b64 in enumerate(data.get("control_previews") or []):
            path = OUT_DIR / f"{stem}-control-{i}.png"
            path.write_bytes(base64.b64decode(b64))
            written.append(path)
    else:
        path = OUT_DIR / f"{stem}.png"
        path.write_bytes(resp.content)
        written.append(path)

    for path in written:
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return written
