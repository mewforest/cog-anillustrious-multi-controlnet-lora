import subprocess
import time
import os

# huggingface_hub's default per-chunk read timeout (10s, via constants.py at
# import time) is too tight for Replicate's internal weights-caching proxy: on
# a cold cache for a given HF repo, the proxy has to fetch-and-relay from the
# Hub itself before the first byte reaches us, which can take longer than 10s
# even though the transfer would otherwise succeed. Must be set before the
# first `import huggingface_hub` anywhere in the process (constants are read
# once at module import), hence this lives at module load time, not inside a
# function.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


class WeightsDownloader:
    @staticmethod
    def download_if_not_exists(url, dest):
        if not os.path.exists(dest):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
        print("downloading took: ", time.time() - start)

    @staticmethod
    def download_hf_snapshot_if_not_exists(repo_id, dest, revision="main"):
        if not os.path.exists(os.path.join(dest, "model_index.json")):
            WeightsDownloader.download_hf_snapshot(repo_id, dest, revision)

    @staticmethod
    def download_hf_snapshot(repo_id, dest, revision="main", max_attempts=4):
        # Public diffusers-format repo (no auth needed): the Civitai download
        # route stopped accepting API-key auth for plain HTTP clients, so the
        # base checkpoint is mirrored on the Hub instead. See README.
        from huggingface_hub import snapshot_download

        start = time.time()
        print("downloading hf snapshot: ", repo_id)
        print("downloading to: ", dest)

        # snapshot_download() already resumes partially-downloaded files on
        # retry (it writes to a .incomplete tmp file and only renames on
        # success), so a plain retry loop is safe and doesn't re-download
        # bytes that already landed -- this just rides out a transient
        # read-timeout/connection blip against Replicate's caching proxy.
        for attempt in range(1, max_attempts + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=dest,
                    allow_patterns=[
                        "model_index.json",
                        "scheduler/*",
                        "text_encoder/*",
                        "text_encoder_2/*",
                        "tokenizer/*",
                        "tokenizer_2/*",
                        "unet/*",
                        "vae/*",
                    ],
                )
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise
                wait = 5 * attempt
                print(
                    f"hf snapshot download attempt {attempt}/{max_attempts} "
                    f"failed ({e!r}), retrying in {wait}s..."
                )
                time.sleep(wait)
        print("downloading took: ", time.time() - start)
