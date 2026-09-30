"""Run a DROID VLA in Isaac Sim (sim-evals) and inspect every decision.

For each policy call ("decision", every `open_loop_horizon` control steps) it logs:
  - the object-level choice distribution estimated from N sampled action chunks
  - which choice was executed and how probable it was under the model
  - token usage (image / text / action) and latency
and draws it as a side panel next to the camera views in the episode video.

Runs inside the Isaac Sim container, in the sim-evals venv, with inspector_server.py running:
    python inspect_eval.py --scene 1 --instruction "put the cube in the bowl" --port 8000
    python inspect_eval.py --tasks-file tasks.json
Outputs in --out-dir/<timestamp>_<task>/: episode_<k>.mp4, decisions.jsonl, summary.json
"""
import argparse
import dataclasses
import json
import pathlib
import re
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inspector_core as ic  # noqa: E402

DEFAULT_INSTRUCTIONS = {1: "put the cube in the bowl", 2: "put the can in the mug", 3: "put banana in the bin"}


@dataclasses.dataclass
class Args:
    scene: int = 1
    instruction: Optional[str] = None  # default: the scene's sim-evals instruction
    tasks_file: Optional[str] = None  # JSON list of {"scene": int, "instruction": str, "episodes": int}
    episodes: int = 1
    host: str = "127.0.0.1"
    port: int = 8000
    open_loop_horizon: int = 8  # control steps executed per decision (sim-evals default)
    max_steps: Optional[int] = None  # default: env episode length (30 s = 450 steps)
    out_dir: str = "results/isaac"
    model_tag: str = "model"
    headless: bool = True
    panel_width: int = 432  # 448 + 432 = 880 px wide (multiple of 16 for video codecs)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


class Scene:
    """Reads robot / object / gripper poses from the Isaac Lab scene, in the robot base frame."""

    def __init__(self, env):
        self.scene = env.unwrapped.scene
        self.robot = self.scene["robot"]
        names = list(self.robot.data.body_names)
        pads = [i for i, n in enumerate(names) if re.search(r"inner_finger|finger_pad|fingertip|finger_tip", n, re.I)]
        if pads:
            self.tcp_bodies, self.tcp_extra = pads, 0.0
        else:  # fall back to the gripper base + a fixed reach along the flange axis
            base = [i for i, n in enumerate(names) if re.search(r"robotiq.*base|gripper.*base|base_link", n, re.I)]
            self.tcp_bodies, self.tcp_extra = base[-1:] or [len(names) - 1], 0.14
        self.objects = sorted(self.scene.rigid_objects.keys())
        print(f"[inspect] tool point from bodies {[names[i] for i in self.tcp_bodies]} (+{self.tcp_extra} m)")
        print(f"[inspect] objects: {self.objects}")

    def _base(self):
        d = self.robot.data
        return d.root_pos_w[0].cpu().numpy(), d.root_quat_w[0].cpu().numpy()

    def tcp(self):
        bp, bq = self._base()
        p = self.robot.data.body_pos_w[0, self.tcp_bodies].cpu().numpy().mean(axis=0)
        return ic.world_to_base(p, bp, bq)

    def object_positions(self):
        bp, bq = self._base()
        return {n: ic.world_to_base(self.scene[n].data.root_pos_w[0].cpu().numpy(), bp, bq) for n in self.objects}


def _write_video(path, frames, fps=15):
    """mediapy needs a system ffmpeg; fall back to imageio's bundled ffmpeg, then OpenCV. Never raises."""
    try:
        import mediapy
        mediapy.write_video(path, frames, fps=fps)
        return path
    except Exception as e1:
        try:
            import imageio
            imageio.mimwrite(path, frames, fps=fps)
            return path
        except Exception as e2:
            try:
                import cv2
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
                return path
            except Exception as e3:
                print(f"[inspect] WARNING video not saved: {e1!r} / {e2!r} / {e3!r}")
                return None


def _request(obs, instruction):
    from openpi_client import image_tools

    pol = obs["policy"]
    ext = pol["external_cam"][0].cpu().numpy()
    wrist = pol["wrist_cam"][0].cpu().numpy()
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(ext, 224, 224),
        "observation/wrist_image_left": image_tools.resize_with_pad(wrist, 224, 224),
        "observation/joint_position": pol["arm_joint_pos"].cpu().numpy(),
        "observation/gripper_position": pol["gripper_pos"].cpu().numpy(),
        "prompt": instruction,
    }


def run_task(env, client, meta, scene_id, instruction, episodes, args, out_root):
    import cv2
    import torch

    out = out_root / f"{datetime.now():%Y%m%d_%H%M%S}_{args.model_tag}_s{scene_id}_{_slug(instruction)}"
    out.mkdir(parents=True, exist_ok=True)
    dec_file = (out / "decisions.jsonl").open("w")
    summaries = []
    for ep in range(episodes):
        obs, _ = env.reset()
        obs, _ = env.reset()  # second render cycle loads materials correctly (sim-evals note)
        sc = Scene(env)
        max_steps = args.max_steps or env.unwrapped.max_episode_length
        frames, plan, k, dec_idx = [], None, 0, 0
        tok_total, lat_total, timeline = 0, 0.0, []
        panel, prev_pred, fk_err = None, None, []
        t_ep = time.time()
        # 48 px tall: 48 + 512 keeps the frame height a multiple of 16 for video codecs
        header = ic.render_header(448 + args.panel_width, meta.get("model", args.model_tag), instruction)
        for step in range(max_steps):
            if plan is None or k >= args.open_loop_horizon:
                req = _request(obs, instruction)
                q_now = req["observation/joint_position"].astype(float)
                g_now = float(np.asarray(req["observation/gripper_position"]).ravel()[0])
                tcp_now = sc.tcp()
                offset = ic.tcp_offset(q_now, tcp_now) + np.array([0, 0, sc.tcp_extra])
                if prev_pred is not None:  # how well FK predicted where the last chunk would end up
                    fk_err.append(float(np.linalg.norm(prev_pred - tcp_now) * 100))
                p0 = ic.tcp_path([q_now], offset)[0]
                resp = client.infer(req)
                plan, k = np.asarray(resp["actions"]), 0
                h = args.open_loop_horizon  # compare samples over the part of the chunk that actually gets executed
                d = ic.decide([s[:h] for s in resp["samples"]], plan[:h], p0, g_now, sc.object_positions(), offset)
                prev_pred = ic.tcp_path(plan[args.open_loop_horizon - 1: args.open_loop_horizon, :7], offset)[0]
                tokens = dict(resp["tokens"])
                tok_total += tokens["image"] + tokens["text"] + tokens["action"]
                tokens["episode_total"] = tok_total
                lat_total += resp["latency_ms"]["policy"]
                rec = {"episode": ep, "decision": dec_idx, "step": step, "instruction": instruction, "scene": scene_id,
                       "model": meta.get("model"), **d, "tokens": tokens, "latency_ms": resp["latency_ms"],
                       "n_samples": int(len(resp["samples"])),
                       "gripper": round(g_now, 3), "tcp": [round(float(v), 4) for v in p0]}
                dec_file.write(json.dumps(rec) + "\n")
                dec_file.flush()
                timeline.append((dec_idx, d["executed"], round(d["p_executed"], 3)))
                print(f"[{args.model_tag}] ep{ep} dec{dec_idx:03d} step{step:03d} exec='{d['executed']}' "
                      f"p={d['p_executed']:.0%} | " + ", ".join(f"{l} {p:.0%}" for l, p in d["choices"][:3]) +
                      f" | tok {tokens['image']}+{tokens['text']}+{tokens['action']} | {resp['latency_ms']['policy']:.0f} ms")
                title = f"Decision {dec_idx} | {meta.get('model', args.model_tag)} | step {step}/{max_steps}"
                dec_idx += 1

            action = plan[k].copy()
            action[-1] = 1.0 if action[-1] > 0.5 else 0.0  # binarize gripper (as sim-evals does)
            k += 1
            obs, _, term, trunc, _ = env.step(torch.tensor(action, dtype=torch.float32)[None])

            pol = obs["policy"]
            views = np.concatenate([
                cv2.resize(pol["external_cam"][0].cpu().numpy(), (448, 256)),
                cv2.resize(pol["wrist_cam"][0].cpu().numpy(), (448, 256))], axis=0)
            panel = ic.render_panel(views.shape[0], args.panel_width, title, d, tokens, resp["latency_ms"],
                                    f"task: {instruction}"[:60])
            frames.append(np.concatenate([header, np.concatenate([views, panel], axis=1)], axis=0))
            if bool(term) or bool(trunc):
                break

        video = out / f"episode_{ep}.mp4"
        _write_video(video, frames, fps=15)
        summaries.append({
            "episode": ep, "steps": step + 1, "decisions": dec_idx, "tokens_total": tok_total,
            "mean_policy_latency_ms": lat_total / max(dec_idx, 1), "wall_s": round(time.time() - t_ep, 1),
            "fk_check_cm": round(float(np.median(fk_err)), 2) if fk_err else None,
            "final_objects": {n: [round(float(v), 3) for v in p] for n, p in sc.object_positions().items()},
            "timeline": timeline, "video": video.name,
        })
        print(f"[inspect] episode {ep}: {dec_idx} decisions, {tok_total:,} tokens, video {video}")
    dec_file.close()
    (out / "summary.json").write_text(json.dumps({"instruction": instruction, "scene": scene_id,
                                                  "server": meta, "episodes": summaries}, indent=2))
    return out


def main(args: Args):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args, _ = parser.parse_known_args([])
    app_args.enable_cameras = True
    app_args.headless = args.headless
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import sim_evals.environments  # noqa: F401  (registers "DROID")
    from isaaclab_tasks.utils import parse_env_cfg
    from openpi_client import websocket_client_policy

    tasks = (json.loads(pathlib.Path(args.tasks_file).read_text()) if args.tasks_file
             else [{"scene": args.scene, "instruction": args.instruction or DEFAULT_INSTRUCTIONS[args.scene],
                    "episodes": args.episodes}])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata()
    print(f"[inspect] server: {meta}")
    out_root = pathlib.Path(args.out_dir)

    scenes = {t["scene"] for t in tasks}
    if len(scenes) > 1:  # Isaac Lab can't reliably rebuild the stage for a new USD in one process
        raise SystemExit(f"one scene per run, got {sorted(scenes)}: split the tasks file or submit one job per scene")
    cfg = parse_env_cfg("DROID", device=app_args.device, num_envs=1, use_fabric=True)
    cfg.set_scene(tasks[0]["scene"])
    env = gym.make("DROID", cfg=cfg)
    for t in tasks:
        out = run_task(env, client, meta, t["scene"], t["instruction"], int(t.get("episodes", 1)), args, out_root)
        print(f"[inspect] wrote {out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
