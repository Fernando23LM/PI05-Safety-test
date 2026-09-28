"""Sample N initial layouts for each custom scene and save them to scenes/<name>.init.npy.

LIBERO's placement sampler is not reproducible from a seed, so we sample once and
store the MuJoCo states; align_eval.py then replays them with env.set_init_state(),
exactly like the stock LIBERO benchmark does. Runs in the LIBERO client venv.

    python make_init_states.py            # all scenes/*.bddl without a .init.npy
    python make_init_states.py --force
"""
import sys

import numpy as np

from scene_utils import SCENE_DIR, make_env

N = 50

if __name__ == "__main__":
    force = "--force" in sys.argv
    for bddl in sorted(SCENE_DIR.glob("*.bddl")):
        out = bddl.with_suffix(".init.npy")
        if out.exists() and not force:
            print("exists:", out.name)
            continue
        env = make_env(bddl, resolution=64)
        states = []
        for i in range(N):
            env.reset()
            states.append(env.get_sim_state().copy())
        env.close()
        np.save(out, np.stack(states))
        print(f"{out.name}: {len(states)} states, dim {states[0].shape[0]}")
