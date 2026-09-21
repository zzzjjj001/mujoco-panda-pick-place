"""脚本化 pick-and-place 策略: 笛卡尔目标 + 有限状态机。

    HOVER -> DESCEND -> CLOSE -> LIFT -> TRANSIT -> LOWER -> RELEASE
    -> RETRACT -> DONE

`act()` 把最新观测 info 映射为一个笛卡尔 TCP 目标 + 夹爪指令；误差持续收敛
或单状态超时则推进状态机。每个状态对应一段确定性控制逻辑。
"""
import numpy as np

from controller.gripper import GRIPPER_HOLD, GRIPPER_OPEN
from controller.ik_solver import quat_mul

# 顶抓目标姿态: 夹爪 z 轴竖直向下, 开口方向沿世界 +x（与参考姿态一致）
TCP_TARGET_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

HOVER_HEIGHT = 0.12      # 悬停高度 (立方体上方)
GRASP_DEPTH = 0.0        # TCP 对准立方体中心: 指垫带关于重心对称,
                         # 重心低于夹持带会形成倒置摆, 闭合时立方体易被挤转
LIFT_HEIGHT = 0.20       # 抬升高度
PLACE_HOVER = 0.15       # 放置区上空高度
PLACE_DROP = 0.06        # 释放高度 (低空释放, 高空自由落体的弹跳会把立方体弹出垫区)

POS_TOL = 0.005          # 位置收敛阈值 [m]
ORI_TOL = 0.05           # 姿态收敛阈值 [rad]
HOLD_STEPS = 3           # 误差需连续保持这么多个控制步
STATE_TIMEOUT = 500      # 单状态超时 (控制步数, 100Hz 下即 5s)
CLOSE_STEPS = 60         # 对准闭合阶段的上限 (0.6s)
OPEN_STEPS = 40          # 等待夹爪张开 (0.4s)
SQUEEZE_STEPS = 30       # 夹住后的挤压稳定等待 (0.3s)
W_STALL_TOL = 0.0005     # 夹爪失速检测: 指垫间隙变化小于该值 [m]
W_STALL_STEPS = 10       # 连续失速这么多控制步 -> 认为已夹住 (0.1s)


class PickPlacePolicy:
    def __init__(self, pad_pos):
        self.pad_pos = np.asarray(pad_pos, float)
        self.state = "HOVER"
        self.done = False
        self._t = 0        # 当前状态已用的控制步
        self._hold = 0     # 连续收敛的控制步数
        self._grasp_pos = None  # 抓取点（DESCEND 结束时冻结的 TCP 位置）
        self._stall = 0    # 夹爪失速计数 (CLOSE 状态)
        self._last_w = None     # 上一步的指垫间隙
        self._stalled = False   # CLOSE 两段式: 已完成对准触碰
        self._yaw = None        # 开口对齐角 (首步冻结, 见 act())

    def _advance(self, state):
        self.state = state
        self._t = 0
        self._hold = 0
        self._stall = 0
        self._last_w = None
        self._stalled = False

    def act(self, info):
        self._t += 1
        if self._yaw is None:
            # 首步冻结开口对齐角: 把夹爪开口转到与立方体表面平行 (折叠到 ±45°,
            # 关节 7 行程充足)。随机偏航下对角夹持的切向分力会吃掉摩擦裕度,
            # 导致搬运中途滑落; 对齐后的面夹持实测可稳定保持整个搬运过程。
            yaw = info.get("cube_yaw", 0.0)
            self._yaw = yaw - np.round(yaw / (np.pi / 2)) * (np.pi / 2)
        # 目标姿态 = 绕世界 z 旋转对齐角 后的顶抓姿态
        quat = quat_mul(
            np.array([np.cos(self._yaw / 2), 0.0, 0.0, np.sin(self._yaw / 2)]),
            TCP_TARGET_QUAT)
        cube = info["cube_pos"]
        tcp = info["tcp_pos"]
        converged = info["err_pos"] < POS_TOL and info["err_ori"] < ORI_TOL
        self._hold = self._hold + 1 if converged else 0
        timeout = self._t > STATE_TIMEOUT

        if self.state == "HOVER":
            # 移动到立方体上方悬停点（目标实时跟踪立方体位置）
            target = cube + np.array([0.0, 0.0, HOVER_HEIGHT])
            grip = GRIPPER_OPEN
            if self._hold >= HOLD_STEPS or timeout:
                self._advance("DESCEND")

        elif self.state == "DESCEND":
            # 垂直下降到抓取高度
            target = cube + np.array([0.0, 0.0, -GRASP_DEPTH])
            grip = GRIPPER_OPEN
            if self._hold >= HOLD_STEPS or timeout:
                self._grasp_pos = tcp.copy()
                self._advance("CLOSE")

        elif self.state == "CLOSE":
            # 闭合夹爪 (GRIPPER_HOLD: 单指 ~1.1N, 闭合中自动把立方体推中),
            # 指垫间隙稳定 (双侧接触) 后再保持挤压片刻, 然后抬升
            target = self._grasp_pos
            grip = GRIPPER_HOLD
            if not self._stalled:
                w = 2.0 * info["finger_q"]
                if self._last_w is not None:
                    self._stall = self._stall + 1 if abs(w - self._last_w) < W_STALL_TOL else 0
                self._last_w = w
                if self._stall >= W_STALL_STEPS or self._t >= CLOSE_STEPS:
                    self._stalled = True
                    self._t = 0
            elif self._t >= SQUEEZE_STEPS:
                self._advance("LIFT")

        elif self.state == "LIFT":
            # 垂直抬升
            target = self._grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
            grip = GRIPPER_HOLD
            if self._hold >= HOLD_STEPS or timeout:
                self._advance("TRANSIT")

        elif self.state == "TRANSIT":
            # 搬运到放置区上空
            target = self.pad_pos + np.array([0.0, 0.0, PLACE_HOVER])
            grip = GRIPPER_HOLD
            if self._hold >= HOLD_STEPS or timeout:
                self._advance("LOWER")

        elif self.state == "LOWER":
            # 垂直下降到低空释放高度
            target = self.pad_pos + np.array([0.0, 0.0, PLACE_DROP])
            grip = GRIPPER_HOLD
            if self._hold >= HOLD_STEPS or timeout:
                self._advance("RELEASE")

        elif self.state == "RELEASE":
            # 张开夹爪放下物体
            target = self.pad_pos + np.array([0.0, 0.0, PLACE_DROP])
            grip = GRIPPER_OPEN
            if self._t >= OPEN_STEPS:
                self._advance("RETRACT")

        elif self.state == "RETRACT":
            # 回撤
            target = self.pad_pos + np.array([0.0, 0.0, PLACE_HOVER + 0.08])
            grip = GRIPPER_OPEN
            if self._hold >= HOLD_STEPS or timeout:
                self.done = True
                self.state = "DONE"

        else:  # DONE: 保持回撤姿态
            target = self.pad_pos + np.array([0.0, 0.0, PLACE_HOVER + 0.08])
            grip = GRIPPER_OPEN

        return target, quat, grip
