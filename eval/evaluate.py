"""批量评估脚本化 pick-and-place 策略，输出可复现的成功率指标。"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from controller.state_machine import PickPlacePolicy
from eval.diagnose import classify_failure

SETTLE_STEPS = 60  # 策略完成后等待物体静止的控制步数 (0.6s)


def run_episode(env, seed, viewer=None, recorder=None, debug=False):
    """运行单个 episode，返回其统计与失败分类。"""
    obs, info = env.reset(seed=seed)
    policy = PickPlacePolicy(env.pad_pos)
    settle = 0
    truncated = False
    while True:
        action = policy.act(info)
        obs, reward, terminated, truncated, info = env.step(*action)
        if viewer is not None:
            viewer.sync()
        if recorder is not None:
            recorder.add(env.data)
        if env.cube_off_table():
            truncated = True  # 立方体被碰落桌面，提前结束
        if debug and env.step_count % 25 == 0:
            print(f"  [{policy.state:8s}] err {info['err_pos']:.3f}/{info['err_ori']:.3f} "
                  f"tcp {np.round(info['tcp_pos'], 3)} cube {np.round(info['cube_pos'], 3)} "
                  f"w {env.gripper.width_m:.3f} lift {info['lift']:.3f}")
        if policy.done:
            settle += 1
        if truncated or settle >= SETTLE_STEPS:
            break

    ep = {
        "steps": env.step_count,
        "grasp_success": bool(env.grasp_success),
        "place_success": env.place_success(),
        "max_lift": round(float(env.max_lift), 4),
        "contact_ever": bool(env.contact_ever),
        "min_tcp_dist": round(float(min(env.min_tcp_dist, 2.0)), 4),
        "cube_off_table": env.cube_off_table(),
        "final_state": policy.state,
        "truncated": bool(truncated),
    }
    ep["failure"] = classify_failure(ep)
    return ep


class EpisodeRecorder:
    """可选的 mp4 录制（mujoco.Renderer + imageio）。"""

    def __init__(self, model, path, camera="eval_cam", fps=50, size=(1280, 720)):
        self._writer = None
        self._renderer = None
        self._camera = camera
        self._frame_every = 2  # 100Hz 控制 -> 50fps
        self._count = 0
        try:
            import imageio
        except ImportError:
            print("[recorder] 未安装 imageio，录制已禁用 (pip install imageio imageio-ffmpeg)")
            return
        try:
            self._writer = imageio.get_writer(str(path), fps=fps)
        except Exception as e:
            print(f"[recorder] 无法创建视频写入器: {e}")
            self._writer = None
            return
        from mujoco import Renderer
        self._renderer = Renderer(model, height=size[1], width=size[0])

    def add(self, data):
        if self._writer is None:
            return
        self._count += 1
        if self._count % self._frame_every:
            return
        self._renderer.update_scene(data, camera=self._camera)
        self._writer.append_data(self._renderer.render())

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def evaluate(env, n_episodes=50, seed=42, viewer=None, recorder=None,
             record_episodes=1, verbose=True, debug=False):
    """运行 N 个 episode 并聚合指标（固定随机种子保证可复现）。"""
    failures = Counter()
    rows = []
    t0 = time.time()
    for ep_idx in range(n_episodes):
        rec = recorder if (recorder is not None and ep_idx < record_episodes) else None
        ep = run_episode(env, seed=seed + ep_idx, viewer=viewer, recorder=rec, debug=debug)
        ep["episode"] = ep_idx
        rows.append(ep)
        if ep["failure"]:
            failures[ep["failure"]] += 1
        if verbose:
            status = ("GRASP+PLACE" if ep["place_success"]
                      else "GRASP" if ep["grasp_success"] else "FAIL")
            line = (f"ep {ep_idx:3d} | {status:12s} | steps {ep['steps']:4d} "
                    f"| max_lift {ep['max_lift']:.3f}")
            if ep["failure"]:
                line += f" | {ep['failure']}"
            print(line)

    n = len(rows)
    grasp_rate = float(np.mean([r["grasp_success"] for r in rows]))
    place_rate = float(np.mean([r["place_success"] for r in rows]))
    success_steps = [r["steps"] for r in rows if r["place_success"]]
    return {
        "n_episodes": n,
        "seed": seed,
        "grasp_success_rate": round(grasp_rate, 4),
        "place_success_rate": round(place_rate, 4),
        "avg_steps_place_success": round(float(np.mean(success_steps)), 1) if success_steps else None,
        "failure_breakdown": dict(failures),
        "wall_time_s": round(time.time() - t0, 1),
        "per_episode": rows,
    }


def save_results(results, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已写入 {p}")
