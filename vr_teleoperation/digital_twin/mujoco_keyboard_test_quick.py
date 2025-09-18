#!/usr/bin/env python3
"""
MuJoCo Keyboard Control Test - 빠른 시작 버전
Enter 없이 바로 시작 (주의: 실물 로봇 위치 먼저 확인!)
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import threading
import sys
import select
import termios
import tty

# MuJoCo 경로
XML_SCENE_PATH = '../dual_arm/scene_dual.xml'

# 오프셋 보정값
LEFT_OFFSETS = [0.0, -0.43, 1.94, -0.42]
RIGHT_OFFSETS = [0.66, -1.03, 0.96, -2.07]

# 조인트 안전 범위
JOINT_LIMITS = {
    'j1': (-1.57, 1.57),
    'j2': (-1.5, 1.5),
    'j3': (-1.5, 1.4),
    'j4': (-1.7, 1.97)
}
GRIPPER_RANGE = (-0.01, 0.019)

# 키보드 제어 설정
JOINT_INCREMENT = 0.05
GRIPPER_INCREMENT = 0.005

class MuJoCoKeyboardController(Node):
    def __init__(self):
        super().__init__('mujoco_keyboard_controller')

        print("🎮 MuJoCo Keyboard Control Test (Quick Start)")
        print("⚠️  실물 로봇이 안전 위치에 있는지 확인하세요!")
        print("📊 목표 초기 자세: J2=-0.3, J3=0.8\n")

        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(XML_SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        # 액추에이터 매핑
        self.left_map = self._map_actuators(side="L")
        self.right_map = self._map_actuators(side="R")

        # 현재 조인트 값
        self.left_joints = [0.0, -0.3, 0.8, 0.0]
        self.right_joints = [0.0, -0.3, 0.8, 0.0]
        self.left_gripper = -0.01
        self.right_gripper = -0.01

        # 선택된 팔과 조인트
        self.selected_arm = 'left'
        self.selected_joint = 0

        # ROS2 퍼블리셔
        self.setup_ros2_publishers()
        self.timer = self.create_timer(0.02, self.publish_to_hardware)

        # 바로 초기화 완료 (빠른 시작)
        self.initialized = True
        self.update_mujoco()

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        print("✅ 준비 완료! 키보드로 제어하세요\n")

    def setup_ros2_publishers(self):
        self.left_traj_pub = self.create_publisher(
            JointTrajectory, '/left_arm_controller/joint_trajectory', 10)
        self.right_traj_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)
        self.joint_state_pub = self.create_publisher(
            JointState, '/mujoco_joint_states', 10)

    def _map_actuators(self, side="L"):
        if side == "L":
            names = {
                "j1": "actuator_joint1",
                "j2": "actuator_joint2",
                "j3": "actuator_joint3",
                "j4": "actuator_joint4",
                "g": "actuator_gripper_joint",
            }
        else:
            names = {
                "j1": "actuator_joint1_r",
                "j2": "actuator_joint2_r",
                "j3": "actuator_joint3_r",
                "j4": "actuator_joint4_r",
                "g": "actuator_gripper_joint_r",
            }

        out = {}
        for k, nm in names.items():
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except:
                aid = -1
            out[k] = aid
        return out

    def update_mujoco(self):
        # 왼팔
        for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
            if self.left_map[k] >= 0:
                self.data.ctrl[self.left_map[k]] = self.left_joints[i]
        if self.left_map['g'] >= 0:
            self.data.ctrl[self.left_map['g']] = self.left_gripper

        # 오른팔
        for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
            if self.right_map[k] >= 0:
                self.data.ctrl[self.right_map[k]] = self.right_joints[i]
        if self.right_map['g'] >= 0:
            self.data.ctrl[self.right_map['g']] = self.right_gripper

    def publish_to_hardware(self):
        if not self.initialized:
            return

        current_time = self.get_clock().now()

        # 왼팔 전송
        left_traj = JointTrajectory()
        left_traj.header.stamp = current_time.to_msg()
        left_traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']

        left_point = JointTrajectoryPoint()
        left_point.positions = [
            self.left_joints[0] + LEFT_OFFSETS[0],
            self.left_joints[1] + LEFT_OFFSETS[1],
            self.left_joints[2] + LEFT_OFFSETS[2],
            self.left_joints[3] + LEFT_OFFSETS[3],
            self.left_gripper
        ]
        left_point.time_from_start = Duration(sec=0, nanosec=100000000)
        left_traj.points = [left_point]
        self.left_traj_pub.publish(left_traj)

        # 오른팔 전송
        right_traj = JointTrajectory()
        right_traj.header.stamp = current_time.to_msg()
        right_traj.joint_names = ['joint1_r', 'joint2_r', 'joint3_r', 'joint4_r', 'gripper_r']

        right_point = JointTrajectoryPoint()
        right_point.positions = [
            self.right_joints[0] + RIGHT_OFFSETS[0],
            self.right_joints[1] + RIGHT_OFFSETS[1],
            self.right_joints[2] + RIGHT_OFFSETS[2],
            self.right_joints[3] + RIGHT_OFFSETS[3],
            self.right_gripper
        ]
        right_point.time_from_start = Duration(sec=0, nanosec=100000000)
        right_traj.points = [right_point]
        self.right_traj_pub.publish(right_traj)

    def handle_key(self, key):
        # 팔 선택
        if key == 'q':
            self.selected_arm = 'left'
            print(f"🖐 왼팔 선택")
        elif key == 'e':
            self.selected_arm = 'right'
            print(f"🖐 오른팔 선택")

        # 조인트 선택
        elif key in '12345':
            self.selected_joint = int(key) - 1
            joint_name = ['J1', 'J2', 'J3', 'J4', 'Gripper'][self.selected_joint]
            print(f"🎯 {self.selected_arm.upper()} {joint_name} 선택")

        # 값 조절
        elif key in '+-':
            increment = JOINT_INCREMENT if key == '+' else -JOINT_INCREMENT

            if self.selected_arm == 'left':
                if self.selected_joint < 4:
                    joint_key = ['j1', 'j2', 'j3', 'j4'][self.selected_joint]
                    lo, hi = JOINT_LIMITS[joint_key]
                    self.left_joints[self.selected_joint] += increment
                    self.left_joints[self.selected_joint] = np.clip(
                        self.left_joints[self.selected_joint], lo, hi)
                else:
                    gripper_inc = GRIPPER_INCREMENT if key == '+' else -GRIPPER_INCREMENT
                    self.left_gripper = np.clip(self.left_gripper + gripper_inc, *GRIPPER_RANGE)
            else:
                if self.selected_joint < 4:
                    joint_key = ['j1', 'j2', 'j3', 'j4'][self.selected_joint]
                    lo, hi = JOINT_LIMITS[joint_key]
                    self.right_joints[self.selected_joint] += increment
                    self.right_joints[self.selected_joint] = np.clip(
                        self.right_joints[self.selected_joint], lo, hi)
                else:
                    gripper_inc = GRIPPER_INCREMENT if key == '+' else -GRIPPER_INCREMENT
                    self.right_gripper = np.clip(self.right_gripper + gripper_inc, *GRIPPER_RANGE)

            self.update_mujoco()
            self.print_status()

        # 리셋
        elif key == 'r':
            self.left_joints = [0.0, -0.3, 0.8, 0.0]
            self.right_joints = [0.0, -0.3, 0.8, 0.0]
            self.update_mujoco()
            print("🔄 초기 자세로 리셋")

        # 도움말
        elif key == 'h':
            print("\n🎮 조작법:")
            print("  Q/E: 왼팔/오른팔 선택")
            print("  1-5: Joint/Gripper 선택")
            print("  +/-: 값 증가/감소")
            print("  R: 리셋, H: 도움말, ESC: 종료")

    def print_status(self):
        print(f"왼팔: [{', '.join([f'{j:.2f}' for j in self.left_joints])}]")
        print(f"오른팔: [{', '.join([f'{j:.2f}' for j in self.right_joints])}]")

def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        return key
    return None

def main():
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setraw(sys.stdin.fileno())
        rclpy.init()

        controller = MuJoCoKeyboardController()

        # ROS2 스피닝 스레드
        ros_thread = threading.Thread(
            target=lambda: rclpy.spin(controller),
            daemon=True
        )
        ros_thread.start()

        print("🎮 키보드 조작:")
        print("  Q/E: 팔 선택, 1-5: 조인트 선택")
        print("  +/-: 조절, R: 리셋, ESC: 종료\n")

        # MuJoCo 뷰어
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

            while viewer.is_running():
                key = get_key()
                if key:
                    if key == '\x1b':  # ESC
                        break
                    elif key in 'qe12345+-rhQER+-RH':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        controller.handle_key(key.lower())
                        tty.setraw(sys.stdin.fileno())

                mujoco.mj_step(controller.model, controller.data)
                viewer.sync()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️ 중단됨")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        rclpy.shutdown()
        print("\n🏁 종료")

if __name__ == "__main__":
    main()