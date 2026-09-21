"""MuJoCo Panda pick-and-place 环境 (Gymnasium 风格 API)。

环境负责物理仿真。一次 `step()` = 一个控制步 (`ctrl_decimation` 个仿真子步，
默认 5 x 2ms，即 100Hz 控制 / 500Hz 仿真)。动作为笛卡尔 TCP 目标 + 原始夹爪
指令；内部由 IK 求解器把笛卡尔目标转换为 7 个臂关节的位置伺服目标。

成功判定（参考 leap-dexterous-grasping 的严格标准）:
  grasp_success : 立方体被抬升到桌面上方 LIFT_THRESHOLD 并持续 SUSTAIN_STEPS 步
  place_success : episode 结束时立方体静止在放置垫范围内
"""
from pathlib import Path

import mujoco
import numpy as np

from controller.ik_solver import DEFAULT_XML, IKSolver
from controller.gripper import GRIPPER_OPEN, GripperController

# 立方体随机生成区域（桌面上, 距基座 r<=0.63, 保证顶抓可达）
CUBE_X_RANGE = (0.38, 0.58)
CUBE_Y_RANGE = (-0.24, 0.24)
CUBE_HALF = 0.025

LIFT_THRESHOLD = 0.05   # 抓取判定: 抬升高度阈值 [m]
SUSTAIN_STEPS = 20      # 需持续的控制步数 (0.2s @100Hz)
PAD_TOL = 0.065         # 放置判定: 立方体中心距垫板中心容差 [m]
PAD_TOP_TOL = 0.03      # 放置判定: 高度容差 [m]
TCP_REF_SPEED = 0.18    # TCP 参考目标移动速度上限 [m/s] (抑制状态切换的加速度冲击)


class PandaGraspEnv:
    def __init__(self, xml_path=None, seed=None, ctrl_decimation=5,
                 max_ctrl_steps=1800, q_ref=None, k_vel_damp=0.35):
        self.k_vel_damp = k_vel_damp  # IK 指令空间速度阻尼系数
        self.xml_path = str(xml_path) if xml_path else str(DEFAULT_XML)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.ctrl_decimation = int(ctrl_decimation)
        self.max_ctrl_steps = int(max_ctrl_steps)
        self.rng = np.random.default_rng(seed)

        m = self.model
        # 场景元素 id / 地址
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        self.cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.cube_jnt = m.body_jntadr[self.cube_body]
        self.cube_qadr = m.jnt_qposadr[self.cube_jnt]
        self.cube_vadr = m.jnt_dofadr[self.cube_jnt]

        pad_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_pad")
        self.pad_pos = m.body_pos[pad_body].copy()
        table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.table_top_z = float(m.geom_pos[table_geom][2] + m.geom_size[table_geom][2])

        # 夹爪相关 geom（用于接触检测）
        self.grip_geoms = set()
        for bname in ("hand", "left_finger", "right_finger"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            gadr, gnum = m.body_geomadr[bid], m.body_geomnum[bid]
            self.grip_geoms.update(range(gadr, gadr + gnum))

        # 7 个臂关节执行器的 ctrl 地址（每个执行器为标量, ctrl 地址 == 执行器 id）
        self.arm_ctrl_adr = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            for i in range(1, 8)
        ]

        self.ik = IKSolver(m, self.data, q_ref=q_ref)
        self.gripper = GripperController(m, self.data)

        # episode 统计（评估与失败诊断用）
        self.step_count = 0
        self.grasp_success = False
        self.max_lift = 0.0
        self.min_tcp_dist = np.inf
        self.contact_ever = False
        self._sustain = 0
        self._last_err = np.zeros(6)
        self._tcp_ref = None  # 速度受限的 TCP 参考目标（斜坡）

        self.reset()

    # ---- gymnasium 风格 API ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # 机械臂从顶抓参考姿态起步
        self.q_ctrl = self.ik.q_ref.copy()
        self.data.qpos[self.ik.qadr] = self.q_ctrl
        self.data.qpos[self.gripper.finger_qadr] = 0.04  # 手指张开

        # 立方体随机位姿（落在桌面上）
        x = self.rng.uniform(*CUBE_X_RANGE)
        y = self.rng.uniform(*CUBE_Y_RANGE)
        yaw = self.rng.uniform(0.0, 2.0 * np.pi)
        qadr = self.cube_qadr
        self.data.qpos[qadr + 0] = x
        self.data.qpos[qadr + 1] = y
        self.data.qpos[qadr + 2] = self.table_top_z + CUBE_HALF + 1e-4
        self.data.qpos[qadr + 3:qadr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self.data.qvel[:] = 0.0

        self.data.ctrl[self.arm_ctrl_adr] = self.q_ctrl
        self.gripper.command(GRIPPER_OPEN)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.grasp_success = False
        self.max_lift = 0.0
        self.min_tcp_dist = np.inf
        self.contact_ever = False
        self._sustain = 0
        self._last_err = np.zeros(6)
        self._tcp_ref = self.ik.tcp_pose()[0].copy()
        return self._obs(), self._info()

    def step(self, target_pos, target_quat, gripper_cmd):
        """一个控制步: 笛卡尔目标 -> IK -> 位置伺服 -> 仿真。"""
        # 参考目标斜坡: 以 TCP_REF_SPEED 限速趋近指令目标,
        # 避免状态切换时目标突变引起的大加速度 (会把夹住的物体晃脱)
        goal = np.asarray(target_pos, float)
        diff = goal - self._tcp_ref
        dist = float(np.linalg.norm(diff))
        max_step = TCP_REF_SPEED * self.model.opt.timestep * self.ctrl_decimation
        if dist > max_step:
            self._tcp_ref = self._tcp_ref + diff * (max_step / dist)
        else:
            self._tcp_ref = goal.copy()

        dq, err = self.ik.solve_step(self._tcp_ref, target_quat, self.q_ctrl)
        # 关节速度阻尼: 抵消伺服滞后引起的超调 (指令空间粘滞项)
        dq -= self.k_vel_damp * self.data.qvel[self.ik.vadr] * (
            self.model.opt.timestep * self.ctrl_decimation)
        self._last_err = err
        self.q_ctrl = np.clip(self.q_ctrl + dq, self.ik.q_lo, self.ik.q_hi)
        self.data.ctrl[self.arm_ctrl_adr] = self.q_ctrl
        self.gripper.command(gripper_cmd)

        for _ in range(self.ctrl_decimation):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # 收敛判据用「TCP -> 指令目标」而非「TCP -> 斜坡参考点」:
        # 斜坡每步只挪 1.8mm, 若用参考点误差则永远小于容差, 状态机会误判收敛
        tcp_now = self.ik.tcp_pose()[0]
        self._last_err[:3] = tcp_now - goal

        info = self._info()
        # 评估用统计
        lift = float(info["cube_pos"][2]) - self.table_top_z
        self.max_lift = max(self.max_lift, lift)
        self._sustain = self._sustain + 1 if lift > LIFT_THRESHOLD else 0
        if self._sustain >= SUSTAIN_STEPS:
            self.grasp_success = True
        self.contact_ever = self.contact_ever or bool(info["contact_now"])
        self.min_tcp_dist = min(self.min_tcp_dist, float(info["tcp_cube_dist"]))

        obs = self._obs()
        reward = self._reward(info)
        terminated = False
        truncated = self.step_count >= self.max_ctrl_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # ---- 评估辅助 ----------------------------------------------------------
    def place_success(self):
        """episode 结束时立方体是否静止在放置垫内。"""
        cube = self.data.xpos[self.cube_body]
        xy_ok = (abs(cube[0] - self.pad_pos[0]) <= PAD_TOL
                 and abs(cube[1] - self.pad_pos[1]) <= PAD_TOL)
        z_ok = cube[2] < self.pad_pos[2] + CUBE_HALF + PAD_TOP_TOL
        vel = np.linalg.norm(self.data.qvel[self.cube_vadr:self.cube_vadr + 6])
        return bool(xy_ok and z_ok and vel < 0.05)

    def cube_off_table(self):
        """立方体是否被碰落桌面。"""
        cube = self.data.xpos[self.cube_body]
        return bool(cube[2] < self.table_top_z - 0.02
                    or abs(cube[0]) > 0.85 or abs(cube[1]) > 0.6)

    # ---- 内部 --------------------------------------------------------------
    def _gripper_contact(self):
        """夹爪 (hand/手指) 与立方体之间是否存在接触。"""
        for c in self.data.contact[:self.data.ncon]:
            g1, g2 = c.geom[0], c.geom[1]
            if (g1 == self.cube_geom and g2 in self.grip_geoms) or \
               (g2 == self.cube_geom and g1 in self.grip_geoms):
                return True
        return False

    def _obs(self):
        return np.concatenate([
            self.data.qpos[self.ik.qadr],          # 关节位置 (7)
            self.data.qvel[self.ik.vadr],          # 关节速度 (7)
            [self.gripper.finger_q],               # 手指开度 (1)
            self.data.xpos[self.cube_body],        # 立方体位置 (3)
            self.ik.tcp_pose()[0],                 # TCP 位置 (3)
        ])

    def _info(self):
        tcp_pos, _ = self.ik.tcp_pose()
        cube_pos = self.data.xpos[self.cube_body].copy()
        return {
            "tcp_pos": tcp_pos,
            "cube_pos": cube_pos,
            "tcp_cube_dist": float(np.linalg.norm(tcp_pos - cube_pos)),
            "err_pos": float(np.linalg.norm(self._last_err[:3])),
            "err_ori": float(np.linalg.norm(self._last_err[3:])),
            "lift": float(cube_pos[2]) - self.table_top_z,
            "grasp_success": self.grasp_success,
            "place_success": self.place_success(),
            "finger_q": self.gripper.finger_q,
            "cube_yaw": float(np.arctan2(self.data.xmat[self.cube_body][3],
                                         self.data.xmat[self.cube_body][0])),
            "contact_now": self._gripper_contact(),
            "step_count": self.step_count,
        }

    def _reward(self, info):
        """密集奖励（如需接入 RL 使用），量级做了归一化。"""
        r = -1.0 * info["tcp_cube_dist"]
        if info["contact_now"]:
            r += 0.5
        if self.gripper.is_gripping():
            r += 1.0
        if self.grasp_success:
            r += 2.0
        if info["place_success"]:
            r += 5.0
        return r
