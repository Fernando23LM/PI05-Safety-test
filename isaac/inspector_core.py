"""Turn sampled VLA action chunks into object-level choices ("reach white bottle 62%").

Pure numpy (+ OpenCV only for drawing), so it can be unit-tested without Isaac Sim.

DROID policies output a chunk of absolute joint positions: (H, 8) = 7 Franka joints + gripper.
We push each sample through Franka forward kinematics to get where the gripper goes, then
compare that path with the known object positions from the simulator:

  reach X          gripper moves >= 1 cm closer to X (and X is the object it approaches most)
  grasp X          ... and the chunk closes the gripper within 6 cm of X
  carry A -> X     already holding A (gripper closed near A), moving toward X
  release A at X   holding A and the chunk opens the gripper
  hold A / no approach   nothing gets meaningfully closer
  undecodable      the model's output could not be decoded into actions (pi0-FAST only)

The probability of a choice is the fraction of the N samples that received that label.
These are heuristics over geometry, not the model's own categories.
"""
import math
from collections import Counter

import numpy as np

# Franka Panda, modified DH (Craig): (a, d, alpha) per joint; flange adds d = 0.107
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
       (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
_FLANGE_D = 0.107

APPROACH_M = 0.01  # must get at least this much closer to count as moving toward an object
GRASP_M = 0.06  # gripper closes within this distance of an object -> "grasp"
HELD_M = 0.08  # gripper closed and object this close -> considered held
PLACE_M = 0.15  # releasing within this distance of another object -> "release A at X"


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1]])


def flange_pose(q):
    """4x4 pose of the Panda flange in the robot base frame for 7 joint angles."""
    T = np.eye(4)
    for (a, d, alpha), th in zip(_DH, q[:7]):
        T = T @ _mdh(a, d, alpha, float(th))
    return T @ _mdh(0.0, _FLANGE_D, 0.0, 0.0)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_to_base(p_w, base_pos, base_quat_wxyz):
    return quat_wxyz_to_R(base_quat_wxyz).T @ (np.asarray(p_w) - np.asarray(base_pos))


def tcp_offset(q_now, tcp_base):
    """Gripper tool point expressed in the flange frame, calibrated from the simulator."""
    T = flange_pose(q_now)
    return T[:3, :3].T @ (np.asarray(tcp_base) - T[:3, 3])


def tcp_path(q_chunk, offset):
    """(H, 3) tool-point positions in the base frame for a chunk of joint positions."""
    out = []
    for q in q_chunk:
        T = flange_pose(q)
        out.append(T[:3, 3] + T[:3, :3] @ offset)
    return np.asarray(out)


def label_chunk(chunk, p0, gripper_now, objects, offset):
    """Object-level label for one action chunk.

    chunk        (H, 8) joint positions + gripper command (>0.5 = close)
    p0           current tool point (base frame)
    gripper_now  current gripper opening in [0, 1] (1 = closed)
    objects      {name: position in base frame}
    """
    chunk = np.asarray(chunk, dtype=float)
    if chunk.size == 0 or not np.all(np.isfinite(chunk)) or np.allclose(chunk, 0):
        return "undecodable", None
    path = tcp_path(chunk[:, :7], offset)
    closes = chunk[:, 7] > 0.5
    names = list(objects)
    if not names:
        return "no objects", path[-1]
    pos = np.stack([objects[n] for n in names])
    d0 = np.linalg.norm(pos - p0, axis=1)
    dmin = np.min(np.linalg.norm(path[:, None, :] - pos[None], axis=2), axis=0)
    dend = np.linalg.norm(pos - path[-1], axis=1)

    held = None
    if gripper_now > 0.5 and d0.min() < HELD_M:
        held = names[int(np.argmin(d0))]
    approach = d0 - dmin
    if held is not None:
        approach[names.index(held)] = -np.inf
    best = int(np.argmax(approach))

    if held is not None:
        if not closes[-1]:
            others = [i for i, n in enumerate(names) if n != held]
            if others:
                j = min(others, key=lambda i: dend[i])
                if dend[j] < PLACE_M:
                    return f"release {held} at {names[j]}", path[-1]
            return f"release {held}", path[-1]
        if approach[best] > APPROACH_M:
            return f"carry {held} -> {names[best]}", path[-1]
        return f"hold {held}", path[-1]
    if approach[best] <= APPROACH_M:
        return "no approach", path[-1]
    if closes.any() and gripper_now <= 0.5 and dmin[best] < GRASP_M:
        return f"grasp {names[best]}", path[-1]
    return f"reach {names[best]}", path[-1]


def decide(samples, executed, p0, gripper_now, objects, offset):
    """Label every sample and the executed chunk; return the choice distribution."""
    labels, ends = [], []
    for s in samples:
        lab, end = label_chunk(s, p0, gripper_now, objects, offset)
        labels.append(lab)
        if end is not None:
            ends.append(end)
    counts = Counter(labels)
    n = max(len(labels), 1)
    choices = sorted(((lab, c / n) for lab, c in counts.items()), key=lambda x: -x[1])
    exec_label, exec_end = label_chunk(executed, p0, gripper_now, objects, offset)
    spread = float(np.mean(np.linalg.norm(np.asarray(ends) - np.mean(ends, axis=0), axis=1)) * 100) if len(ends) > 1 else 0.0
    return {
        "choices": choices,  # [(label, probability)], most likely first
        "executed": exec_label,
        "p_executed": counts.get(exec_label, 0) / n,  # how typical the executed choice is under the model
        "entropy_bits": float(-sum(p * math.log2(p) for _, p in choices if p > 0)),
        "spread_cm": spread,  # how far apart the sampled end points are
        "executed_end": None if exec_end is None else [round(float(v), 4) for v in exec_end],
    }


# --------------------------------------------------------------------------- drawing
def render_header(width, model, instruction, height=48):
    """Top bar for the video: model name, then the instruction it was given. Returns uint8 RGB."""
    import cv2

    img = np.full((height, width, 3), (12, 18, 30), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def fit(s, scale):  # shorten with an ellipsis until it fits the bar
        while len(s) > 4 and cv2.getTextSize(s, font, scale, 1)[0][0] > width - 28:
            s = s[:-4] + "..."
        return s

    cv2.putText(img, fit(f"Model: {model}", 0.55), (14, 20), font, 0.55, (90, 170, 255), 1, cv2.LINE_AA)
    cv2.putText(img, fit(f'Instruction: "{instruction}"', 0.55), (14, 40), font, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def render_panel(height, width, title, decision, tokens, latency, step_info):
    """Dark side panel in the style of an agent 'current state' card. Returns uint8 RGB."""
    import cv2

    img = np.full((height, width, 3), (22, 30, 46), np.uint8)
    white, grey, accent, bar_bg = (235, 235, 235), (150, 160, 175), (90, 170, 255), (45, 55, 75)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 26

    def text(s, color=white, scale=0.5, thick=1, dy=22):
        nonlocal y
        cv2.putText(img, s, (14, y), font, scale, color, thick, cv2.LINE_AA)
        y += dy

    text(title, grey, 0.45)
    text(f"Executed: {decision['executed']}", white, 0.55, 1, 26)
    text(f"p(executed) = {decision['p_executed']:.0%}   entropy {decision['entropy_bits']:.2f} bits", grey, 0.42)
    text(f"sample spread {decision['spread_cm']:.1f} cm", grey, 0.42, dy=28)
    text("Choice distribution", grey, 0.45)
    bar_w = width - 28
    for lab, p in decision["choices"][:5]:
        cv2.rectangle(img, (14, y - 12), (14 + bar_w, y + 6), bar_bg, -1)
        cv2.rectangle(img, (14, y - 12), (14 + int(bar_w * p), y + 6), accent, -1)
        cv2.putText(img, f"{p:4.0%}  {lab}"[:48], (18, y + 2), font, 0.42, white, 1, cv2.LINE_AA)
        y += 24
    y += 8
    text("Token usage (this decision)", grey, 0.45)
    text(f"image {tokens['image']}  text {tokens['text']}  action {tokens['action']}", white, 0.45)
    if tokens.get("denoise_steps"):
        text(f"({tokens['action']} action tokens x {tokens['denoise_steps']} denoising passes)", grey, 0.4)
    text(f"episode total: {tokens.get('episode_total', 0):,} tokens", grey, 0.42, dy=28)
    text("Latency", grey, 0.45)
    text(f"policy {latency['policy']:.0f} ms   +{latency['samples']:.0f} ms for samples", white, 0.45, dy=28)
    text(step_info, grey, 0.42)
    return img
