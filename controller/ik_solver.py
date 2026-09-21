"""Franka Panda 阻尼最小二乘 (DLS) 逆运动学。

menagerie 的 Panda 模型由高增益位置伺服驱动 (actuator1..7)，因此 IK 以
"分辨率速率" (resolved-rate) 增量模式使用: 每个控制步计算一个小的关节增量
dq，用于减小 TCP（两指尖之间的点）与笛卡尔目标之间的 6 维误差；积分后的
关节目标发送给位置执行器，从而得到平滑运动。

TCP 位于 hand 坐标系原点沿其局部 z 轴 0.103 m 处（指尖垫中间高度）。
menagerie XML 没有在该点定义 site，因此点雅可比由 body 雅可比加叉乘项得到:
    J_point = J_origin - skew(r) @ J_angular
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# hand 坐标系原点 -> 指尖垫中间的抓取点
TCP_OFFSET = np.array([0.0, 0.0, 0.103])

# 顶抓参考姿态（menagerie home 姿态即为夹爪竖直向下的顶抓构型），
# 用作 IK 零空间偏置与 episode 初始姿态
DEFAULT_Q_REF = np.array([0.0, 0.0, 0.0, -1.5707963, 0.0, 1.5707963, -0.7853982])

# 与 DEFAULT_Q_REF 对应的 TCP 目标姿态（home 姿态下 hand 的四元数）:
# z 轴竖直向下, 夹爪开口方向沿世界 +x
DEFAULT_TCP_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "franka_emika_panda" / "grasp_scene.xml"


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两个 (w, x, y, z) 四元数的哈密顿积。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_error_world(q_cur: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    """世界系旋转矢量: 将姿态 q_cur 转到 q_tgt 所需的轴角。

    世界系角速度对应左乘旋转: q_tgt = exp(w) * q_cur
    => q_err = q_tgt * conj(q_cur)，再取最短路径转为轴角。"""
    q_inv = q_cur * np.array([1.0, -1.0, -1.0, -1.0])
    q_err = quat_mul(q_tgt, q_inv)
    if q_err[0] < 0.0:  # 取最短路径
        q_err = -q_err
    v = np.linalg.norm(q_err[1:])
    if v < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(v, q_err[0])
    return q_err[1:] / v * angle


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class IKSolver:
    """带零空间姿态偏置的 6 维阻尼最小二乘 IK。"""

    def __init__(self, model, data, damping=0.08, max_dq=0.03, k_null=0.05,
                 k_pos=0.35, k_ori=0.5, q_ref=None):
        self.model, self.data = model, data
        self.q_ref = DEFAULT_Q_REF.copy() if q_ref is None else np.asarray(q_ref, float)

        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                        for n in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jnt_ids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.jnt_ids])
        self.q_lo = model.jnt_range[self.jnt_ids, 0].copy()
        self.q_hi = model.jnt_range[self.jnt_ids, 1].copy()

        self.damping = damping      # DLS 阻尼系数 lambda
        self.max_dq = max_dq        # 每控制步关节增量上限 [rad]
        self.k_null = k_null        # 零空间姿态偏置增益
        self.k_pos = k_pos          # 位置误差任务增益 (<1 抑制物理闭环振荡)
        self.k_ori = k_ori          # 姿态误差任务增益

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---- 运动学辅助 ------------------------------------------------------
    def tcp_pose(self):
        """返回 TCP 世界坐标 (3,) 与 hand 旋转矩阵 (3, 3)。"""
        R = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        pos = self.data.xpos[self.hand_id] + R @ TCP_OFFSET
        return pos, R

    def tcp_quat(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self.hand_id])
        return quat

    def _task_jacobian(self, R):
        """6x7 任务雅可比 [J_point; J_angular]，仅取 7 个臂关节列。"""
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.hand_id)
        jacp = self._jacp[:, self.vadr]
        jacr = self._jacr[:, self.vadr]
        r = R @ TCP_OFFSET
        return np.vstack([jacp - skew(r) @ jacr, jacr])

    # ---- 每控制步一次的分辨率速率解 --------------------------------------
    def solve_step(self, target_pos, target_quat, q_arm, q_ref=None):
        """返回 (dq, err)。err = [位置误差(3), 姿态误差(3)]。"""
        tcp_pos, R = self.tcp_pose()
        quat = self.tcp_quat()

        pos_err = np.asarray(target_pos, float) - tcp_pos
        ori_err = quat_error_world(quat, np.asarray(target_quat, float))
        err = np.concatenate([pos_err, ori_err])

        J = self._task_jacobian(R)
        JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))

        # 缩放后的任务增量（全量 Newton 步在物理闭环中易振荡，取分数增益）
        task = np.concatenate([self.k_pos * pos_err, self.k_ori * ori_err])
        dq = J_pinv @ task
        # 零空间: 在不影响末端任务的前提下偏向参考姿态（避奇异/避限位）
        q_ref = self.q_ref if q_ref is None else np.asarray(q_ref, float)
        N = np.eye(7) - J_pinv @ J
        dq += N @ (self.k_null * (q_ref - q_arm))

        dq = np.clip(dq, -self.max_dq, self.max_dq)
        return dq, err


if __name__ == "__main__":
    # 姿态探测 + 全工作区 IK 收敛测试（纯运动学，不涉及物理）
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    ik = IKSolver(model, data)
    pos, R = ik.tcp_pose()
    print("home TCP:", np.round(pos, 3), "quat:", np.round(ik.tcp_quat(), 4),
          "z-axis:", np.round(R[:, 2], 3))

    targets = {
        "center-hover": [0.5, 0.0, 0.585],
        "far-corner": [0.60, 0.25, 0.585],
        "near-corner": [0.40, -0.25, 0.585],
        "pad-hover": [0.42, -0.36, 0.58],
        "low-grasp": [0.5, 0.0, 0.44],
        "far-low": [0.62, -0.20, 0.44],
        "lift-high": [0.5, 0.0, 0.65],
    }
    for tname, tgt in targets.items():
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        solver = IKSolver(model, data)
        q_arm = data.qpos[solver.qadr].copy()
        err = np.zeros(6)
        for i in range(300):
            dq, err = solver.solve_step(np.array(tgt), DEFAULT_TCP_QUAT, q_arm)
            q_arm = np.clip(q_arm + dq, solver.q_lo, solver.q_hi)
            data.qpos[solver.qadr] = q_arm
            mujoco.mj_forward(model, data)
            if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
                break
        ok = "OK " if (np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2) else "BAD"
        print(f"{ok} {tname:13s}: iters={i:3d} pos_err={np.linalg.norm(err[:3]):.4f} "
              f"ori_err={np.linalg.norm(err[3:]):.4f} q={np.round(q_arm, 2)}")
