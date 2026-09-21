"""腱驱 Panda 夹爪控制辅助。

actuator8 是 `split` 腱上的位置伺服（menagerie 原始设计）:
    ctrl 255 -> 手指关节 0.04 (完全张开)
    ctrl 0   -> 手指关节 0.0  (完全闭合)

腱力 F = 0.015686*ctrl - 100*q - 10*qdot (正值张开, 腱系数 0.5/0.5,
即每根手指分到 0.5*F 的夹紧力)。指垫接触面在手指坐标系 y=0 处,
故两指垫间隙 = 2*q_finger [m]。

夹住 5cm 立方体时 q_pinch = 0.025, 单指夹紧力 N = 0.5*(100*q - 0.015686*ctrl):
    ctrl=0   -> N=1.25 N
    ctrl=24  -> N=1.06 N  <- 闭合/保持共用 (可克服桌面摩擦自动对中)
    ctrl=156 -> N=0.02 N  (近乎零的触碰力)
    ctrl=159 -> N=0     (恰好贴住, 无夹紧力)
"""
import mujoco
import numpy as np

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
GRIPPER_RANGE = (GRIPPER_CLOSED, GRIPPER_OPEN)

# 闭合与保持共用指令: 单指夹紧力 ~1.1 N (q_pinch=0.025 时 N = 0.5*(2.5-0.376))。
# 该力略大于立方体与桌面间的摩擦力 (~1N), 闭合时会自动把立方体推到两指中间
# 对中 (实测 w 收敛到 0.050 的标准面夹持); 对中后总摩擦 ~2.2N ≈ 2.2 倍重力。
# 实测: 更温和的指令 (ctrl=156, 0.14N) 推不动立方体, 会造成单侧接触假夹持;
# ctrl=0 (1.25N) 与 24 差别不大, 但 24 的保持更柔和。
GRIPPER_HOLD = 24.0

# 手指垫贴住 5cm 立方体两侧时的手指关节值 (间隙 = 2*q = 0.05 m)
CUBE_PINCH_FINGER_Q = 0.025


class GripperController:
    def __init__(self, model, data, actuator="actuator8"):
        self.model, self.data = model, data
        # 每个执行器为标量, ctrl 地址 == 执行器 id
        self.ctrl_adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        self.finger_qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("finger_joint1", "finger_joint2")
        ]

    def command(self, value):
        """发送原始伺服目标 (255 = 张开, 0 = 闭合)。"""
        self.data.ctrl[self.ctrl_adr] = float(np.clip(value, *GRIPPER_RANGE))

    @property
    def finger_q(self):
        return float(self.data.qpos[self.finger_qadr[0]])

    @property
    def width_m(self):
        """当前指垫间隙 [m]。"""
        return 2.0 * self.finger_q

    def is_gripping(self):
        """手指停在行程中段（约立方体宽度附近）-> 正在捏住物体。"""
        return 0.4 * CUBE_PINCH_FINGER_Q < self.finger_q < 1.4 * CUBE_PINCH_FINGER_Q
