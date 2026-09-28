"""Scene / environment helpers shared by align_eval.py, dump_frames.py and make_init_states.py.

Two kinds of scenes:
  libero_goal        stock LIBERO-Goal scene; initial states come from LIBERO's own files
                     (cond["init"] names the LIBERO-Goal task whose 50 states are reused)
  scenes/<name>.bddl custom hazard scenes; initial states are pre-generated once by
                     make_init_states.py into scenes/<name>.init.npy, so every prompt in a
                     scene starts from exactly the same 50 layouts.
"""
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SCENE_DIR = pathlib.Path(__file__).resolve().parent / "scenes"
RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def bddl_and_states(scene, init_task=None):
    """Return (bddl_path, init_states or None) for a scene."""
    if scene == "libero_goal":
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        ids = {suite.get_task(i).name: i for i in range(suite.n_tasks)}
        tid = ids[init_task]
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        return bddl, suite.get_task_init_states(tid)
    bddl = SCENE_DIR / f"{scene}.bddl"
    init = SCENE_DIR / f"{scene}.init.npy"
    return bddl, (np.load(init) if init.exists() else None)


def make_env(bddl, seed=7, resolution=RESOLUTION):
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def env_key(cond):
    return cond["scene"] if cond["scene"] != "libero_goal" else "libero_goal/" + cond["init"]
