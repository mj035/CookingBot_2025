#!/usr/bin/env python3
"""
🤖 Dual-Arm VR → Physical Robot Bridge (초기 자세 캘리브레이션)
MuJoCo [0,0,0,0]과 실물 로봇 초기 자세를 매칭
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import socket
import json
import numpy as np
import threading
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand

class DualArmCalibratedMirror(Node):
    def __init__(self):
        super().__init__('dual_arm_calibrated_mirror')

        print("\n🤖 양팔 미러링 - 캘리브레이션 방식")
        print("📍 초기 자세 맞추기:")
        print("   1. 메타퀘스트를 MuJoCo [0,0,0,0] 자세로")
        print("   2. 실물 로봇도 동일한 자세로")
        print("   3. 미러링 시작\n")

        # 오프셋 값 (Wizard 측정 기반)
        # MuJoCo [0,0,0,0]일 때 실물이 가져야 할 값
        self.HARDWARE_ZERO_POSE = {
            'left': [0.034, 0.376, 1.853, -0.176],
            'right': [0.634, -0.887, 0.841, -2.163]
        }

        print("📊 MuJoCo [0,0,0,0]에 해당하는 하드웨어 값:")
        print(f"   왼팔:  {self.HARDWARE_ZERO_POSE['left']}")
        print(f"   오른팔: {self.HARDWARE_ZERO_POSE['right']}\n")

        # 초기값 저장
        self.robot_initial = {
            'left': None,   # 실물 초기 위치
            'right': None
        }

        self.mujoco_initial = {
            'left': None,   # MuJoCo 초기 위치 (보통 [0,0,0,0])
            'right': None
        }

        self.mujoco_current = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }

        self.gripper_values = {
            'left': -0.01,
            'right': -0.01
        }

        # 캘리브레이션 오차
        self.calibration_offset = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }

        # 스무딩용
        self.last_left_joints = None
        self.last_right_joints = None

        self.last_gripper_values = {
            'left': -0.01,
            'right': -0.01
        }

        # 초기화 상태
        self.left_ready = False
        self.right_ready = False

        # Thread safety를 위한 Lock 추가
        self.data_lock = threading.Lock()

        # ROS2 Publishers
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Joint State Subscriber
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # Gripper Action Clients
        self.left_gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(
            self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # MuJoCo 연결
        self.setup_socket()

        # 제어 루프
        self.timer = self.create_timer(0.02, self.dual_arm_control)  # 50Hz

        # 상태 모니터링
        self.status_timer = self.create_timer(3.0, self.print_status)
        self.control_count = 0

        print("🤖 실물 로봇 초기 위치 읽는 중...")

    def joint_states_callback(self, msg):
        """Joint States 콜백"""
        # 왼팔 초기값
        if self.robot_initial['left'] is None:
            left_joints = []
            for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                if name in msg.name:
                    idx = msg.name.index(name)
                    left_joints.append(msg.position[idx])

            if len(left_joints) == 4:
                self.robot_initial['left'] = left_joints
                # 캘리브레이션 오차 계산
                for i in range(4):
                    expected = self.HARDWARE_ZERO_POSE['left'][i]
                    actual = left_joints[i]
                    self.calibration_offset['left'][i] = actual - expected

                print(f"✅ 왼팔 초기값: {[f'{x:.3f}' for x in left_joints]}")
                print(f"   캘리브레이션 오차: {[f'{x:.3f}' for x in self.calibration_offset['left']]}")

                if max(abs(x) for x in self.calibration_offset['left']) > 0.3:
                    print("⚠️  왼팔 초기 자세가 목표와 차이가 큽니다!")

                self.last_left_joints = left_joints.copy()
                self.left_ready = True

        # 오른팔 초기값
        if self.robot_initial['right'] is None:
            right_joints = []
            for name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
                if name in msg.name:
                    idx = msg.name.index(name)
                    right_joints.append(msg.position[idx])

            if len(right_joints) == 4:
                self.robot_initial['right'] = right_joints
                # 캘리브레이션 오차 계산
                for i in range(4):
                    expected = self.HARDWARE_ZERO_POSE['right'][i]
                    actual = right_joints[i]
                    self.calibration_offset['right'][i] = actual - expected

                print(f"✅ 오른팔 초기값: {[f'{x:.3f}' for x in right_joints]}")
                print(f"   캘리브레이션 오차: {[f'{x:.3f}' for x in self.calibration_offset['right']]}")

                if max(abs(x) for x in self.calibration_offset['right']) > 0.3:
                    print("⚠️  오른팔 초기 자세가 목표와 차이가 큽니다!")

                self.last_right_joints = right_joints.copy()
                self.right_ready = True

    def setup_socket(self):
        """MuJoCo 소켓 연결"""
        def recv():
            while True:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('localhost', 12345))
                    sock.settimeout(0.1)
                    print("🔗 MuJoCo 연결됨")

                    buffer = ""
                    # first 플래그 제거 - 각 팔 독립적 초기화
                    while True:
                        try:
                            data = sock.recv(4096).decode('utf-8')
                            if not data:
                                break
                            buffer += data
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                if line:
                                    try:
                                        d = json.loads(line)

                                        # Thread safety를 위해 Lock 사용
                                        with self.data_lock:
                                            # 왼팔 데이터
                                            if 'left_arm' in d and 'joint_angles' in d['left_arm']:
                                                self.mujoco_current['left'] = d['left_arm']['joint_angles'][:4]
                                                if first and self.mujoco_initial['left'] is None:
                                                    self.mujoco_initial['left'] = self.mujoco_current['left'].copy()
                                                    print(f"✅ MuJoCo 왼팔 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")

                                            if 'left_arm' in d and 'gripper' in d['left_arm']:
                                                self.gripper_values['left'] = d['left_arm']['gripper']

                                            # 오른팔 데이터
                                            if 'right_arm' in d and 'joint_angles' in d['right_arm']:
                                                self.mujoco_current['right'] = d['right_arm']['joint_angles'][:4]
                                                if first and self.mujoco_initial['right'] is None:
                                                    self.mujoco_initial['right'] = self.mujoco_current['right'].copy()
                                                    print(f"✅ MuJoCo 오른팔 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['right']]}")

                                                # MuJoCo 초기값 확인
                                                if (max(abs(x) for x in self.mujoco_initial['left']) > 0.1 or
                                                    max(abs(x) for x in self.mujoco_initial['right']) > 0.1):
                                                    print("⚠️  MuJoCo 초기값이 [0,0,0,0]이 아닙니다! VR 캘리브레이션을 확인하세요.")

                                                    first = False

                                            if 'right_arm' in d and 'gripper' in d['right_arm']:
                                                self.gripper_values['right'] = d['right_arm']['gripper']

                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    print(f"⚠️ 연결 오류: {e}")
                    time.sleep(2)

        threading.Thread(target=recv, daemon=True).start()

    def apply_safety_and_smoothing(self, target_joints, arm_side='left'):
        """안전 제한 및 스무딩"""
        # 하드웨어 한계
        joint_limits = [
            [-1.57, 1.57],
            [-1.5, 1.5],
            [-1.5, 1.4],
            [-1.7, 1.97]
        ]

        last_joints = self.last_left_joints if arm_side == 'left' else self.last_right_joints
        if last_joints is None:
            return target_joints

        safe_joints = []
        for i, (target, limits) in enumerate(zip(target_joints, joint_limits)):
            # 급격한 변화 제한
            max_change = 0.15
            change = target - last_joints[i]
            if abs(change) > max_change:
                target = last_joints[i] + np.sign(change) * max_change

            # 스무딩
            smoothed = last_joints[i] * 0.8 + target * 0.2

            # 범위 제한
            safe_joint = np.clip(smoothed, limits[0], limits[1])
            safe_joints.append(safe_joint)

        # 업데이트
        if arm_side == 'left':
            self.last_left_joints = safe_joints.copy()
        else:
            self.last_right_joints = safe_joints.copy()

        return safe_joints

    def create_joint_trajectory(self, target_joints, arm_side='left'):
        """조인트 궤적 생성"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()

        if arm_side == 'right':
            traj.joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
        else:
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = Duration(sec=0, nanosec=50000000)

        traj.points = [point]
        return traj

    def send_gripper_goal(self, position, arm_side='left'):
        """그리퍼 제어"""
        client = self.left_gripper_client if arm_side == 'left' else self.right_gripper_client

        if not client.wait_for_server(timeout_sec=0.1):
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)
        goal_msg.command.max_effort = 100.0

        client.send_goal_async(goal_msg)

    def dual_arm_control(self):
        """양팔 제어 (캘리브레이션 적용)"""
        # Thread safety를 위해 Lock 사용하여 데이터 읽기
        with self.data_lock:
            mujoco_left = self.mujoco_current['left'].copy()
            mujoco_right = self.mujoco_current['right'].copy()
            gripper_left = self.gripper_values['left']
            gripper_right = self.gripper_values['right']

        # 왼팔 제어
        if self.left_ready and self.mujoco_initial['left'] is not None:
            # MuJoCo 변화량 계산
            left_target = []
            for i in range(4):
                # MuJoCo 변화량
                mujoco_delta = mujoco_left[i] - self.mujoco_initial['left'][i]

                # 올바른 제어 공식: 이상적인 초기값 + MuJoCo 변화량
                target = self.HARDWARE_ZERO_POSE['left'][i] + mujoco_delta
                left_target.append(target)

            # 안전 및 스무딩
            safe_left = self.apply_safety_and_smoothing(left_target, 'left')

            # 전송
            left_traj = self.create_joint_trajectory(safe_left, 'left')
            self.left_joint_pub.publish(left_traj)

            # 그리퍼
            if abs(gripper_left - self.last_gripper_values['left']) > 0.002:
                self.send_gripper_goal(gripper_left, 'left')
                self.last_gripper_values['left'] = gripper_left

        # 오른팔 제어
        if self.right_ready and self.mujoco_initial['right'] is not None:
            # MuJoCo 변화량 계산
            right_target = []
            for i in range(4):
                # MuJoCo 변화량
                mujoco_delta = mujoco_right[i] - self.mujoco_initial['right'][i]

                # 올바른 제어 공식: 이상적인 초기값 + MuJoCo 변화량
                target = self.HARDWARE_ZERO_POSE['right'][i] + mujoco_delta
                right_target.append(target)

            # 안전 및 스무딩
            safe_right = self.apply_safety_and_smoothing(right_target, 'right')

            # 전송
            right_traj = self.create_joint_trajectory(safe_right, 'right')
            self.right_joint_pub.publish(right_traj)

            # 그리퍼
            if abs(gripper_right - self.last_gripper_values['right']) > 0.002:
                self.send_gripper_goal(gripper_right, 'right')
                self.last_gripper_values['right'] = gripper_right

        self.control_count += 1

    def print_status(self):
        """상태 출력"""
        print(f"\n📊 === 미러링 상태 ===")
        print(f"✅ 준비: 왼팔={self.left_ready} 오른팔={self.right_ready}")
        print(f"📈 제어 횟수: {self.control_count}")

        if self.left_ready:
            print(f"🎮 왼팔 캘리브레이션 오차: {[f'{x:.3f}' for x in self.calibration_offset['left']]}")

        if self.right_ready:
            print(f"🎮 오른팔 캘리브레이션 오차: {[f'{x:.3f}' for x in self.calibration_offset['right']]}")

        if self.mujoco_initial['left'] and self.mujoco_initial['right']:
            print(f"📍 MuJoCo 초기: L={[f'{x:.2f}' for x in self.mujoco_initial['left']]}, "
                  f"R={[f'{x:.2f}' for x in self.mujoco_initial['right']]}")

        print(f"🖐 그리퍼: L={self.gripper_values['left']:.3f}, R={self.gripper_values['right']:.3f}")

    def emergency_stop(self):
        """비상 정지"""
        print("🛑 비상 정지!")
        if self.last_left_joints:
            self.left_joint_pub.publish(
                self.create_joint_trajectory(self.last_left_joints, 'left'))
        if self.last_right_joints:
            self.right_joint_pub.publish(
                self.create_joint_trajectory(self.last_right_joints, 'right'))

def main():
    rclpy.init()

    try:
        mirror = DualArmCalibratedMirror()

        print("\n🤖 === 양팔 캘리브레이션 미러링 ===")
        print("📍 초기 자세 확인:")
        print("   1. VR 컨트롤러 캘리브레이션 (MuJoCo [0,0,0,0])")
        print("   2. 실물 로봇 초기 자세 맞추기")
        print("   3. 자동으로 오차 보정 후 시작")
        print("🛑 Ctrl+C: 비상 정지\n")

        rclpy.spin(mirror)

    except KeyboardInterrupt:
        print("\n🛑 중단됨")
        if 'mirror' in locals():
            mirror.emergency_stop()
    finally:
        rclpy.shutdown()
        print("🏁 종료")

if __name__ == '__main__':
    main()