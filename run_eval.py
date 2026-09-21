"""入口: 评估脚本化 pick-and-place 控制器。

用法示例:
    python run_eval.py                          # 50 episodes, seed 42, 无头模式
    python run_eval.py --episodes 100 --seed 0
    python run_eval.py --viewer                 # 交互窗口实时观看
    python run_eval.py --record 1               # 保存 results/eval_video.mp4
    python run_eval.py --episodes 1 --debug     # 打印前 1 个 episode 的过程日志
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from env.grasp_env import PandaGraspEnv
from eval.evaluate import EpisodeRecorder, evaluate, save_results


def main():
    p = argparse.ArgumentParser(description="MuJoCo Panda 抓取任务评估")
    p.add_argument("--xml", default=None, help="场景 xml 路径 (默认 franka_emika_panda/grasp_scene.xml)")
    p.add_argument("--episodes", type=int, default=50, help="评估 episode 数")
    p.add_argument("--seed", type=int, default=42, help="随机种子 (保证可复现)")
    p.add_argument("--max-steps", type=int, default=1800, help="每 episode 最大控制步数 (100Hz)")
    p.add_argument("--viewer", action="store_true", help="交互窗口实时观看")
    p.add_argument("--record", type=int, default=0, metavar="N", help="录制前 N 个 episode 为 mp4")
    p.add_argument("--debug", action="store_true", help="打印第一个 episode 的过程日志")
    p.add_argument("--out", default="results/success_rate.json", help="结果 JSON 输出路径")
    args = p.parse_args()

    env = PandaGraspEnv(xml_path=args.xml or None, seed=args.seed,
                        max_ctrl_steps=args.max_steps)
    viewer = None
    recorder = None
    if args.record:
        recorder = EpisodeRecorder(env.model, ROOT / "results" / "eval_video.mp4")
    try:
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        results = evaluate(env, n_episodes=args.episodes, seed=args.seed,
                           viewer=viewer, recorder=recorder,
                           record_episodes=max(args.record, 1), debug=args.debug)
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()

    print("\n===== evaluation summary =====")
    print(f"episodes             : {results['n_episodes']}")
    print(f"seed                 : {results['seed']}")
    print(f"grasp success rate   : {results['grasp_success_rate'] * 100:.1f}%")
    print(f"place  success rate  : {results['place_success_rate'] * 100:.1f}%")
    if results["avg_steps_place_success"] is not None:
        print(f"avg steps (placed)   : {results['avg_steps_place_success']}")
    print(f"failure breakdown    : {results['failure_breakdown']}")
    print(f"wall time            : {results['wall_time_s']}s")

    out = Path(args.out)
    save_results(results, out if out.is_absolute() else ROOT / out)


if __name__ == "__main__":
    main()
