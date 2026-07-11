from collections import deque
import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Some LoRA hosts (Civitai in particular) have been observed serving an HTML
# login/redirect page instead of the file for requests that look like a bare
# script/bot, even with a valid API token. A realistic browser UA + explicit
# redirect following + a content sanity-check make this robust for any host
# (Civitai, HuggingFace, direct links), not just Civitai.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


class WeightsDownloadCache:
    # Canonical filename used when the downloaded weights turn out to be a
    # single-file community LoRA/LoCon rather than a Replicate-trainer bundle.
    # Keeping the .safetensors extension matters: diffusers' load_lora_weights
    # picks its loader based on the weight_name's extension.
    COMMUNITY_LORA_FILENAME = "weights.safetensors"

    def __init__(
        self,
        min_disk_free: int = 10 * (2**30),
        base_dir: str = "/src/weights-cache",
        volume=None,
    ):
        """
        WeightsDownloadCache is meant to track and download weights files as fast
        as possible, while ensuring there's enough disk space.

        It tries to keep the most recently used weights files in the cache, so
        ensure you call ensure() on the weights each time you use them.

        It will not re-download weights files that are already in the cache.

        :param min_disk_free: Minimum disk space required to start download, in bytes.
        :param base_dir: The base directory to store weights files.
        :param volume: Optional object with `.commit()`/`.reload()` methods (e.g. a
            `modal.Volume`) backing `base_dir`. When set, the cache is treated as
            shared across containers: `ensure()` reloads before checking for a hit
            and commits after a download, so a LoRA fetched by one container is
            visible to the next one without re-downloading. Left as `None` (the
            default), this class behaves exactly as it always has -- a plain local
            directory private to one container -- so nothing changes for Cog/Replicate.
        """
        self.min_disk_free = min_disk_free
        self.base_dir = base_dir
        self.volume = volume
        self._hits = 0
        self._misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _remove_least_recent(self) -> None:
        """
        Remove the least recently used weights file from the cache and disk.
        """
        oldest = self.lru_paths.popleft()
        self._rm_disk(oldest)

    def cache_info(self) -> str:
        """
        Get cache information.

        :return: Cache information.
        """

        return f"CacheInfo(hits={self._hits}, misses={self._misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _rm_disk(self, path: str) -> None:
        """
        Remove a weights file or directory from disk.
        :param path: Path to remove.
        """
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def _has_enough_space(self) -> bool:
        """
        Check if there's enough disk space.

        :return: True if there's more than min_disk_free free, False otherwise.
        """
        disk_usage = shutil.disk_usage(self.base_dir)
        print(f"Free disk space: {disk_usage.free}")
        return disk_usage.free >= self.min_disk_free

    def ensure(self, url: str) -> str:
        """
        Ensure weights file is in the cache and return its path.

        This also updates the LRU cache to mark the weights as recently used.

        :param url: URL to download weights file from, if not in cache.
        :return: Path to weights.
        """
        path = self.weights_path(url)

        if self.volume is not None:
            # Pull in whatever other containers have committed since we last
            # looked, so a LoRA another container already fetched shows up
            # as a hit here instead of being downloaded again.
            self.volume.reload()

        if path in self.lru_paths:
            # here we remove to re-add to the end of the LRU (marking it as recently used)
            self._hits += 1
            self.lru_paths.remove(path)
        elif os.path.exists(path):
            # Already materialized on disk (e.g. fetched by another container
            # sharing this volume) even though this process's in-memory LRU
            # doesn't know about it yet -- adopt it instead of re-downloading.
            self._hits += 1
        else:
            self._misses += 1
            self.download_weights(url, path)
            if self.volume is not None:
                self.volume.commit()

        self.lru_paths.append(path)  # Add file to end of cache
        return path

    def weights_path(self, url: str) -> str:
        """
        Generate path to store a weights file based hash of the URL.

        :param url: URL to download weights file from.
        :return: Path to store weights file.
        """
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]  # Use the first 16 characters of the hash
        return os.path.join(self.base_dir, short_hash)

    def download_weights(self, url: str, dest: str) -> None:
        """
        Download weights file from a URL, ensuring there's enough disk space.

        :param url: URL to download weights file from.
        :param dest: Path to store weights file.
        """
        print("Ensuring enough disk space...")
        while not self._has_enough_space() and len(self.lru_paths) > 0:
            self._remove_least_recent()

        print(f"Downloading weights: {url}")

        st = time.time()
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.base_dir, prefix=".download-")
        os.close(tmp_fd)
        try:
            self._download_to_file(url, tmp_path)
            self._materialize(tmp_path, dest)
        except Exception:
            # Never leave a partially-written/half-extracted cache entry behind.
            self._rm_disk(dest)
            raise
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        print(f"Downloaded weights in {time.time() - st} seconds")

    @staticmethod
    def _download_to_file(url: str, tmp_path: str) -> None:
        session = _make_session()
        with session.get(url, stream=True, timeout=(10, 300), allow_redirects=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(
                    f"Expected a weights file from {url} but the server returned "
                    f"an HTML page (Content-Type: {content_type}). This usually "
                    "means the host redirected to a login/error page -- check the "
                    "URL and, for Civitai links, that the API token is valid and "
                    "the model version/file isn't gated."
                )
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Belt-and-suspenders: reject HTML bodies that slipped through without
        # an honest Content-Type header.
        with open(tmp_path, "rb") as f:
            head = f.read(256).lstrip().lower()
        if head.startswith(b"<") and (b"<html" in head or b"<!doctype" in head):
            raise RuntimeError(
                f"Downloaded file from {url} looks like an HTML page, not a "
                "weights file -- check the URL and API token."
            )
        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"Downloaded file from {url} is empty.")

    @staticmethod
    def _materialize(tmp_path: str, dest: str) -> None:
        """Turn the downloaded temp file into `dest` (always a directory):
        either extracted archive contents (old Replicate-trainer bundle) or a
        single community LoRA file under a canonical name."""
        if zipfile.is_zipfile(tmp_path):
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(dest)
        elif tarfile.is_tarfile(tmp_path):
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(tmp_path) as tf:
                tf.extractall(dest)
        elif WeightsDownloadCache._looks_like_safetensors(tmp_path):
            os.makedirs(dest, exist_ok=True)
            shutil.move(
                tmp_path, os.path.join(dest, WeightsDownloadCache.COMMUNITY_LORA_FILENAME)
            )
        else:
            raise RuntimeError(
                "Downloaded weights file is neither a recognized archive "
                "(zip/tar) nor a valid .safetensors file -- the download may "
                "be corrupt, or the URL doesn't point to a real weights file."
            )

    @staticmethod
    def _looks_like_safetensors(path: str) -> bool:
        """A safetensors file starts with an 8-byte little-endian header
        length, followed by that many bytes of a JSON header starting with
        '{'. This is enough to distinguish it from an HTML page or garbage."""
        try:
            with open(path, "rb") as f:
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    return False
                header_len = int.from_bytes(header_len_bytes, "little")
                if header_len <= 0 or header_len > 100 * 1024 * 1024:
                    return False
                header = f.read(min(header_len, 4096)).lstrip()
                return header[:1] == b"{"
        except OSError:
            return False
