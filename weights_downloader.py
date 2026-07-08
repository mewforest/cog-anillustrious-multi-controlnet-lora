import subprocess
import time
import os


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
    def download_hf_snapshot(repo_id, dest, revision="main"):
        # Public diffusers-format repo (no auth needed): the Civitai download
        # route stopped accepting API-key auth for plain HTTP clients, so the
        # base checkpoint is mirrored on the Hub instead. See README.
        from huggingface_hub import snapshot_download

        start = time.time()
        print("downloading hf snapshot: ", repo_id)
        print("downloading to: ", dest)
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
        print("downloading took: ", time.time() - start)
