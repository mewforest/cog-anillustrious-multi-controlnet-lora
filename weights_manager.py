import json
import os

try:
    # Still present as of diffusers 0.31 (pinned for the Modal deployment),
    # but diffusers has been folding per-variant attention processors into a
    # unified one, so a future bump could drop this class. Guarded so
    # importing this module doesn't hard-fail if that happens -- only the
    # legacy Replicate-trainer bundle path (rare, superseded by community
    # LoRA loading) needs it.
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0
except ImportError:
    LoRAAttnProcessor2_0 = None
from safetensors.torch import load_file

from dataset_and_utils import TokenEmbeddingsHandler
from weights import WeightsDownloadCache


class UnsupportedLoraError(Exception):
    """Raised when a LoRA file downloads fine but diffusers can't actually
    apply it -- e.g. true LyCORIS (LoHa/LoKr) algorithms, which use a
    different key layout than standard LoRA/LoCon and aren't supported."""


class WeightsManager:
    def __init__(self, predictor, weights_cache=None):
        self.predictor = predictor
        self.weights_cache = weights_cache if weights_cache is not None else WeightsDownloadCache()
        self._native_lora_loaded = False  # True iff pipe.load_lora_weights() applied something

    def load_trained_weights(self, weights, pipe):
        # weights can be a URLPath, which behaves in unexpected ways
        weights = str(weights)
        if self.predictor.tuned_weights == weights:
            print("skipping loading .. weights already loaded")
            return

        # Whatever was active before is about to change one way or another --
        # clear it up front so a failure below can never leave stale state.
        self._unload_current(pipe)

        try:
            local_weights_cache = self.weights_cache.ensure(weights)

            bundle_marker_files = ("unet.safetensors", "lora.safetensors")
            is_bundle = any(
                os.path.exists(os.path.join(local_weights_cache, f))
                for f in bundle_marker_files
            )
            community_path = os.path.join(
                local_weights_cache, WeightsDownloadCache.COMMUNITY_LORA_FILENAME
            )

            if is_bundle:
                self._load_replicate_trainer_bundle(local_weights_cache, pipe)
            elif os.path.exists(community_path):
                self._load_community_lora(
                    local_weights_cache, WeightsDownloadCache.COMMUNITY_LORA_FILENAME, pipe
                )
            else:
                raise RuntimeError(
                    f"Downloaded weights at {local_weights_cache} don't match "
                    "any recognized format (Replicate-trainer bundle or "
                    "single-file LoRA)."
                )
        except Exception:
            # No silent success: fully reset so the next call -- even for the
            # same URL -- genuinely retries instead of no-op'ing.
            self.predictor.tuned_weights = None
            self.predictor.tuned_model = False
            self.predictor.is_lora = False
            self._native_lora_loaded = False
            raise

        self.predictor.tuned_weights = weights

    def unload_trained_weights(self, pipe):
        """Called when a prediction doesn't request lora_weights but a
        previous prediction on this warm container loaded some -- prevents a
        stale LoRA leaking into an unrelated generation."""
        if self.predictor.tuned_weights is None:
            return
        print("No lora_weights passed for this prediction -- unloading previous weights")
        self._unload_current(pipe)
        self.predictor.tuned_weights = None
        self.predictor.tuned_model = False
        self.predictor.is_lora = False

    def _unload_current(self, pipe):
        if self._native_lora_loaded:
            print("Unloading previous LoRA weights")
            pipe.unload_lora_weights()
            self._native_lora_loaded = False
        elif self.predictor.tuned_weights:
            # Pre-existing limitation inherited from upstream fofr/cog-sdxl,
            # not introduced by this fix: a full UNet fine-tune
            # (unet.safetensors) or the hand-rolled attention-processor LoRA
            # patch (lora.safetensors bundle) mutates the UNet in place and
            # was never designed to be reversible without reloading the base
            # checkpoint from disk. Surface this loudly instead of pretending
            # to unload it.
            print(
                "WARNING: previously loaded weights were a Replicate-trainer "
                "bundle, which cannot be cleanly unloaded on a warm "
                "container -- the base checkpoint will not be restored "
                "until the container is recycled."
            )

    def _load_community_lora(self, directory, filename, pipe):
        print(f"Loading community LoRA/LoCon weights: {filename}")
        try:
            pipe.load_lora_weights(directory, weight_name=filename)
        except Exception as e:
            raise self._wrap_lora_error(e) from e

        # load_lora_weights() can succeed without raising yet apply nothing if
        # the file's key layout doesn't match what diffusers' Kohya converter
        # expects (this is how LyCORIS LoHa/LoKr silently no-ops in this
        # diffusers version). Detect that and turn it into a clear error.
        applied = any(
            "lora" in type(proc).__name__.lower()
            for proc in pipe.unet.attn_processors.values()
        )
        if not applied:
            pipe.unload_lora_weights()
            raise UnsupportedLoraError(
                "load_lora_weights() completed but no LoRA attention layers "
                "were applied to the UNet. This usually means the file uses "
                "an unsupported algorithm (e.g. LyCORIS LoHa/LoKr) -- only "
                "standard LoRA/LoCon files are supported. (Note: this check "
                "only inspects UNet attention layers; a text-encoder-only "
                "LoRA, which is rare for SDXL, would not be caught here.)"
            )

        self._native_lora_loaded = True
        self.predictor.is_lora = True
        self.predictor.tuned_model = False

    @staticmethod
    def _wrap_lora_error(e):
        message = str(e)
        lowered = message.lower()
        hints = ("loha", "lokr", "hada_", "unexpected key", "unrecognized", "unknown lora")
        if any(h in lowered for h in hints):
            return UnsupportedLoraError(
                f"This LoRA file could not be loaded ({message}). It may use "
                "an unsupported algorithm (e.g. LyCORIS LoHa/LoKr) -- only "
                "standard LoRA/LoCon files are supported."
            )
        return e

    def _load_replicate_trainer_bundle(self, local_weights_cache, pipe):
        from no_init import no_init_or_tensor

        print("Loading fine-tuned model")
        self.predictor.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.predictor.is_lora = True

        if not self.predictor.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(maybe_unet_path)
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            if LoRAAttnProcessor2_0 is None:
                raise RuntimeError(
                    "This diffusers version no longer exposes "
                    "LoRAAttnProcessor2_0, so legacy Replicate-trainer LoRA "
                    "bundles (lora.safetensors) can't be loaded here. Use a "
                    "standard community LoRA .safetensors file instead."
                )

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map[name],
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.predictor.token_map = params

        self.predictor.tuned_model = True
