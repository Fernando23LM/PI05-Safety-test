"""First-run check for Isaac Sim on a Gilbreth node (no policy needed).

  - starts Isaac Sim headless and builds the sim-evals DROID scene
  - saves the external + wrist camera views to --out
  - lists the scene's objects and the robot's bodies/joints
  - checks inspector_core's Franka kinematics against the simulator (flange = panda_link8)

    python isaac_check.py --scene 1 --out results/isaac/check
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inspector_core as ic  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--out", default="results/isaac/check")
    args, _ = p.parse_known_args()

    t0 = time.time()
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args, _ = ap.parse_known_args([])
    app_args.enable_cameras, app_args.headless = True, True
    app = AppLauncher(app_args).app
    print(f"[check] Isaac Sim started in {time.time() - t0:.0f} s")

    import cv2
    import gymnasium as gym
    import sim_evals.environments  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg("DROID", device=app_args.device, num_envs=1, use_fabric=True)
    cfg.set_scene(args.scene)
    env = gym.make("DROID", cfg=cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()
    print(f"[check] scene {args.scene} ready after {time.time() - t0:.0f} s")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in ["external_cam", "external_cam_2", "wrist_cam"]:
        img = obs["policy"][cam][0].cpu().numpy()
        cv2.imwrite(str(out / f"scene{args.scene}_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[check] {cam}: {img.shape} {img.dtype}, mean pixel {img.mean():.1f} (0 = black render)")

    scene = env.unwrapped.scene
    robot = scene["robot"]
    print("[check] objects:", {n: scene[n].data.root_pos_w[0].cpu().numpy().round(3).tolist() for n in scene.rigid_objects})
    print("[check] bodies:", list(robot.data.body_names))
    print("[check] joints:", list(robot.data.joint_names))

    # kinematics check at the reset pose and after a few random joint targets
    names = list(robot.data.body_names)
    link = "panda_link8" if "panda_link8" in names else ("panda_link7" if "panda_link7" in names else None)
    back = 0.0 if link == "panda_link8" else 0.107  # link7 frame sits 0.107 m before the flange
    if link is None:
        print("[check] WARNING: no panda_link7/8 body; cannot verify FK directly")
    else:
        li = names.index(link)
        print(f"[check] verifying FK against {link}")
        jidx = [robot.data.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        errs = []
        rng = np.random.default_rng(0)
        for trial in range(4):
            if trial:
                q_target = robot.data.joint_pos[0, jidx].cpu().numpy() + rng.normal(0, 0.15, 7)
                for _ in range(20):
                    obs, *_ = env.step(torch.tensor(np.r_[q_target, 0.0], dtype=torch.float32)[None])
            q = robot.data.joint_pos[0, jidx].cpu().numpy()
            bp, bq = robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy()
            sim = ic.world_to_base(robot.data.body_pos_w[0, li].cpu().numpy(), bp, bq)
            T = ic.flange_pose(q)
            fk = T[:3, 3] - T[:3, :3] @ np.array([0.0, 0.0, back])
            errs.append(np.linalg.norm(sim - fk) * 100)
            print(f"[check] q={q.round(2).tolist()} sim flange={sim.round(3).tolist()} fk={fk.round(3).tolist()} err={errs[-1]:.2f} cm")
        verdict = "OK" if max(errs) < 1.0 else "MISMATCH: object labels will be wrong, report this"
        print(f"[check] FK max error {max(errs):.2f} cm -> {verdict}")
    env.close()
    app.close()
    print(f"[check] done in {time.time() - t0:.0f} s; images in {out}")


if __name__ == "__main__":
    main()
