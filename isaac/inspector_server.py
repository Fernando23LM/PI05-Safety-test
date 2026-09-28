"""openpi policy server that also returns what the inspector needs.

Per request it returns, on top of the normal "actions":
  samples      (N, H, D) extra action chunks for the same observation, used to estimate the
               probability of each object-level choice. pi0.5 (flow matching): N independent
               samples from new noise. pi0-FAST: N samples at `--temperature` (1.0 = the model's
               own token distribution).
  tokens       what the model consumed/produced for this decision: image tokens (256 per
               224x224 camera image), text tokens (instruction + discretized state + template),
               action tokens (pi0.5: action-horizon suffix tokens, re-run for each denoising
               step; pi0-FAST: generated tokens up to end-of-sequence)
  latency_ms   policy = one normal inference; samples = the extra batch (inspection overhead)

The executed chunk is exactly what the plain openpi server would return (pi0.5: one random
sample; pi0-FAST: greedy decoding), so inspecting does not change the robot's behaviour.

Runs in the openpi env (JAX):
    uv run python inspector_server.py --config pi05_droid_jointpos_polaris \
        --checkpoint gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris --port 8000
"""
import dataclasses
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from openpi.models import model as _model
from openpi.models import pi0_fast as _pi0_fast
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi_client import base_policy

TOKENS_PER_IMAGE = 256  # SigLIP So400m/14 at 224x224 -> 16x16 patches
EOS = 1  # PaliGemma end-of-sequence token


@dataclasses.dataclass
class Args:
    config: str = "pi05_droid_jointpos_polaris"
    checkpoint: str = "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
    port: int = 8000
    n_samples: int = 16
    temperature: float = 1.0  # pi0-FAST sampling temperature for the distribution (ignored for pi0.5)
    seed: int = 0


class Inspector(base_policy.BasePolicy):
    def __init__(self, args: Args):
        cfg = _config.get_config(args.config)
        self.policy = policy_config.create_trained_policy(cfg, args.checkpoint)
        self.fast = isinstance(cfg.model, _pi0_fast.Pi0FASTConfig)
        self.horizon = cfg.model.action_horizon
        self.n = args.n_samples
        self.temperature = args.temperature
        self.rng = jax.random.key(args.seed)
        self.sp = _tokenizer.PaligemmaTokenizer()._tokenizer
        self.name = args.config
        logging.info("loaded %s (%s), %d samples per decision", args.config, "pi0-FAST" if self.fast else "flow", self.n)

    def _run(self, inputs, batch, **kw):
        """Batched sample_actions on already-transformed inputs; returns raw model outputs."""
        b = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (batch, *np.shape(x))), inputs)
        obs = _model.Observation.from_dict(b)
        self.rng, key = jax.random.split(self.rng)
        return np.asarray(self.policy._sample_actions(key, obs, **kw))

    def _to_actions(self, state, raw):
        try:
            out = self.policy._output_transform({"state": np.asarray(state), "actions": np.asarray(raw)})
            a = np.asarray(out["actions"], dtype=np.float32)
            return a if np.all(np.isfinite(a)) else None
        except Exception:  # pi0-FAST: sampled tokens that don't decode into a valid action chunk
            return None

    def infer(self, obs: dict) -> dict:
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, obs))
        state = inputs["state"]

        t0 = time.monotonic()
        greedy_kw = {"temperature": 0.0} if self.fast else {}
        raw_exec = self._run(inputs, 1, **greedy_kw)[0]
        t1 = time.monotonic()
        sample_kw = {"temperature": self.temperature} if self.fast else {}
        raw_samples = self._run(inputs, self.n, **sample_kw) if self.n > 0 else np.zeros((0,))
        t2 = time.monotonic()

        executed = self._to_actions(state, raw_exec)
        shape = executed.shape if executed is not None else (self.horizon, 8)
        if executed is None:  # keep the robot still rather than crash
            executed = np.tile(np.asarray(obs["observation/joint_position"], np.float32).tolist() + [0.0], (self.horizon, 1))
        samples = []
        for r in raw_samples:
            a = self._to_actions(state, r)
            samples.append(a if a is not None and a.shape == shape else np.zeros(shape, np.float32))

        img_keys = list(inputs["image"])
        n_img = sum(bool(np.asarray(inputs["image_mask"][k])) for k in img_keys)
        if self.fast:
            toks = np.asarray(raw_exec).astype(int)
            eos = np.flatnonzero(toks == EOS)
            action_tokens, denoise = int(eos[0] + 1 if eos.size else np.count_nonzero(toks)), 0
        else:
            action_tokens, denoise = int(self.horizon), 10
        tokens = {
            "image": TOKENS_PER_IMAGE * n_img,
            "image_padding": TOKENS_PER_IMAGE * (len(img_keys) - n_img),  # masked camera slots
            "text": int(np.sum(inputs["tokenized_prompt_mask"])),
            "instruction": len(self.sp.encode(str(obs.get("prompt", "")))),
            "action": action_tokens,
            "denoise_steps": denoise,
        }
        return {
            "actions": executed,
            "samples": np.stack(samples) if samples else np.zeros((0, *shape), np.float32),
            "tokens": tokens,
            "latency_ms": {"policy": (t1 - t0) * 1000, "samples": (t2 - t1) * 1000},
            "model": self.name,
        }

    @property
    def metadata(self):
        return {"model": self.name, "n_samples": self.n, "temperature": self.temperature if self.fast else None,
                "fast": self.fast, "action_horizon": self.horizon}


def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    policy = Inspector(args)
    websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port,
                                                  metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
