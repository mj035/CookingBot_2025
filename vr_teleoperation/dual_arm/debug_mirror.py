#!/usr/bin/env python3
"""
미러 코드 디버깅 - mirror_dual.py와 동일 동작 + 로그만 추가
기존 코드 그대로, 왼팔 첫 제어 시점 로그만 출력
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import socket
import json
import numpy as np
import threading
import time

class DebugMirror(Node):
    def __init__(self):
        super().__init__('debug_mirror')

        print("\n🔍 === 미러 디버깅 모드 ===")

        # 초기값 저장
        self.robot_initial = {
            'left': None,
            'right': None
        }

        self.mujoco_initial = {
            'left': None,
            'right': None
        }

        self.mujoco_current = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }

        self.first_control = {
            'left': True,
            'right': True
        }

        # Joint limits
        self.joint_limits = [
            [-3.14, 3.14],   # Joint 1
            [-1.5, 1.5],     # Joint 2
            [-1.5, 1.4],     # Joint 3
            [-1.7, 1.97]     # Joint 4
        ]

        # Publishers
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # Socket
        self.setup_socket()

        # Control timer
        self.timer = self.create_timer(0.05, self.debug_control)

        print("📡 초기값 수집 중...")

    def joint_states_callback(self, msg):
        """초기값 저장"""
        # 왼팔
        if self.robot_initial['left'] is None:
            self.robot_initial['left'] = []

            left_indices = {
                'joint1': 7,
                'joint2': 8,
                'joint3': 1,
                'joint4': 5
            }

            for joint_name in ['joint1', 'joint2', 'joint3', 'joint4']:
                if joint_name in left_indices:
                    idx = left_indices[joint_name]
                    if idx < len(msg.position):
                        value = msg.position[idx]
                        self.robot_initial['left'].append(value)

            if len(self.robot_initial['left']) == 4:
                print(f"✅ 왼팔 하드웨어 초기값: {[f'{x:.3f}' for x in self.robot_initial['left']]}")

                # Joint3 체크
                if self.robot_initial['left'][2] > 1.4:
                    print(f"⚠️  왼팔 joint3 범위 초과: {self.robot_initial['left'][2]:.3f} > 1.4")

        # 오른팔
        if self.robot_initial['right'] is None:
            self.robot_initial['right'] = []

            right_indices = {
                'right_joint1': 2,
                'right_joint2': 4,
                'right_joint3': 3,
                'right_joint4': 0
            }

            for joint_name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
                if joint_name in right_indices:
                    idx = right_indices[joint_name]
                    if idx < len(msg.position):
                        value = msg.position[idx]
                        self.robot_initial['right'].append(value)

            if len(self.robot_initial['right']) == 4:
                print(f"✅ 오른팔 하드웨어 초기값: {[f'{x:.3f}' for x in self.robot_initial['right']]}")

    def setup_socket(self):
        """소켓 연결"""
        def recv():
            while True:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('localhost', 12345))
                    sock.settimeout(0.1)
                    print("🔗 브릿지 연결")

                    buffer = ""
                    first = True
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

                                        # 왼팔
                                        if 'left_arm' in d and 'joint_angles' in d['left_arm']:
                                            self.mujoco_current['left'] = d['left_arm']['joint_angles'][:4]

                                            if first and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'].copy()
                                                print(f"✅ 왼팔 MuJoCo 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")

                                        # 오른팔
                                        if 'right_arm' in d and 'joint_angles' in d['right_arm']:
                                            self.mujoco_current['right'] = d['right_arm']['joint_angles'][:4]

                                            if first and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'].copy()
                                                print(f"✅ 오른팔 MuJoCo 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['right']]}")
                                                first = False

                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    time.sleep(2)

        threading.Thread(target=recv, daemon=True).start()

    def apply_safety_limits(self, joints, arm_side='left'):
        """안전 제한 적용 (mirror_dual.py와 동일)"""
        joint_limits = [
            [-3.14, 3.14],   # Joint 1
            [-1.5, 1.5],     # Joint 2
            [-1.5, 1.4],     # Joint 3
            [-1.7, 1.97]     # Joint 4
        ]

        safe_joints = []
        for i, (joint_val, limits) in enumerate(zip(joints, joint_limits)):
            # 급격한 변화 제한 (0.1 라디안/스텝)
            max_change = 0.1
            if hasattr(self, f'last_{arm_side}_joints'):
                last_joints = getattr(self, f'last_{arm_side}_joints')
                if len(last_joints) > i:
                    change = joint_val - last_joints[i]
                    if abs(change) > max_change:
                        joint_val = last_joints[i] + np.sign(change) * max_change

            # 조인트 범위 제한
            safe_joint = np.clip(joint_val, limits[0], limits[1])
            safe_joints.append(safe_joint)

        # 현재 조인트 저장
        setattr(self, f'last_{arm_side}_joints', safe_joints.copy())
        return safe_joints

    def create_joint_trajectory(self, target_joints, arm_side='left'):
        """조인트 궤적 메시지 생성 (mirror_dual.py와 동일)"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()

        if arm_side == 'right':
            traj.joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
        else:
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms

        traj.points = [point]
        return traj

    def debug_control(self):
        """디버깅 제어 (mirror_dual.py의 dual_arm_control과 동일)"""
        # 왼팔 제어
        if (self.robot_initial['left'] is not None and
            self.mujoco_initial['left'] is not None):

            # 오프셋 계산
            left_target = []
            for i in range(4):
                delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                target_val = self.robot_initial['left'][i] + delta
                left_target.append(target_val)

            # 첫 제어 시 로그
            if self.first_control['left']:
                print("\n" + "=" * 60)
                print("🎯 왼팔 첫 제어 시점 분석:")
                print(f"  하드웨어 초기: {[f'{x:.3f}' for x in self.robot_initial['left']]}")
                print(f"  MuJoCo 초기: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")
                print(f"  MuJoCo 현재: {[f'{x:.3f}' for x in self.mujoco_current['left']]}")
                delta_values = []
                for i in range(4):
                    delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                    delta_values.append(f'{delta:.3f}')
                print(f"  변화량: {delta_values}")
                print(f"  목표값(클리핑 전): {[f'{x:.3f}' for x in left_target]}")

                # 안전 제한 체크
                clipped_target = []
                for i, val in enumerate(left_target):
                    clipped = np.clip(val, self.joint_limits[i][0], self.joint_limits[i][1])
                    clipped_target.append(clipped)
                    if abs(val - clipped) > 0.01:
                        print(f"  ⚠️ Joint{i+1} 클리핑: {val:.3f} → {clipped:.3f}")

                print(f"  목표값(클리핑 후): {[f'{x:.3f}' for x in clipped_target]}")

                # 움직임 예상
                movement = []
                for i in range(4):
                    move = clipped_target[i] - self.robot_initial['left'][i]
                    movement.append(move)
                    if abs(move) > 0.1:
                        print(f"  ⚠️ Joint{i+1} 큰 움직임 예상: {move:.3f} rad")

                print("=" * 60)
                self.first_control['left'] = False

            # 기존 코드와 동일하게 안전 제한 적용
            safe_left_target = self.apply_safety_limits(left_target, 'left')

            # 궤적 생성 및 전송
            left_traj = self.create_joint_trajectory(safe_left_target, 'left')
            self.left_joint_pub.publish(left_traj)

        # 오른팔 제어 (mirror_dual.py와 완전히 동일, 로그 없음)
        if (self.robot_initial['right'] is not None and
            self.mujoco_initial['right'] is not None):

            right_target = []
            for i in range(4):
                delta = self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                target_val = self.robot_initial['right'][i] + delta
                right_target.append(target_val)

            # 안전 제한 적용
            safe_right_target = self.apply_safety_limits(right_target, 'right')

            # 궤적 생성 및 전송
            right_traj = self.create_joint_trajectory(safe_right_target, 'right')
            self.right_joint_pub.publish(right_traj)

def main():
    rclpy.init()

    try:
        debug = DebugMirror()

        print("\n🔍 === 미러 디버깅 실행 중 ===")
        print("첫 제어 명령 시점을 분석합니다")
        print("VR 컨트롤러를 캘리브레이션하세요")

        rclpy.spin(debug)

    except KeyboardInterrupt:
        print("\n디버깅 종료")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()