#!/usr/bin/env python3
"""
🤖 Dual-Arm VR → Physical Robot Bridge (No Limits Version)
단일팔처럼 리밋 없이 순수 오프셋만 적용
- MuJoCo에서만 범위 체크 (이중 제한 방지)
- 초기 자세 그대로 유지
- 부드러운 움직임 보장
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

class DualArmNoLimitsMirror(Node):
    def __init__(self):
        super().__init__('dual_arm_no_limits_mirror')

        print("\n" + "="*70)
        print("🤖 DUAL ARM NO LIMITS MIRROR")
        print("단일팔처럼 리밋 없이 순수 오프셋만 적용")
        print("="*70 + "\n")

        # === 초기값 저장 ===
        self.robot_initial = {
            'left': None,
            'right': None
        }

        self.robot_current = {
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

        # === 디버그 플래그 ===
        self._first_control = {
            'left': True,
            'right': True
        }
        self._name_vec = None  # joint_states 순서 추적

        # === Publishers ===
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # === Subscriber ===
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # === Socket setup ===
        self.setup_socket()

        # === Control timer ===
        self.timer = self.create_timer(0.05, self.dual_arm_control)  # 20Hz

        # === Status timer ===
        self.status_timer = self.create_timer(2.0, self.print_status)

        print("📡 하드웨어 초기값 수집 중...")
        print("🎮 VR 브릿지 연결 대기 중...")

    def joint_states_callback(self, msg):
        """Joint States 콜백 - 이름 기반 매핑"""
        # 이름 순서 변화 감지
        if self._name_vec is None:
            self._name_vec = list(msg.name)
            print(f"\n📋 Joint States 구조: {len(msg.name)}개 조인트")
            print(f"  이름: {msg.name}")
        elif self._name_vec != list(msg.name):
            self._name_vec = list(msg.name)
            print("\n⚠️  /joint_states 순서 변경 감지! 재매핑...")

        # 이름 → 인덱스 매핑
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # 왼팔 조인트
        LEFT = ['joint1', 'joint2', 'joint3', 'joint4']
        # 오른팔 조인트
        RIGHT = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']

        # === 왼팔 초기값 (한 번만) ===
        if all(n in name_to_idx for n in LEFT):
            self.robot_current['left'] = [msg.position[name_to_idx[n]] for n in LEFT]

            if self.robot_initial['left'] is None:
                self.robot_initial['left'] = self.robot_current['left'][:]
                print(f"✅ 왼팔 하드웨어 초기값: {self._fmt(self.robot_initial['left'])}")

                # 초기값 분석 (정보 제공용, 제한하지 않음)
                for i, val in enumerate(self.robot_initial['left']):
                    if val > 1.5 or val < -1.5:
                        print(f"  ℹ️  Joint{i+1}: {val:.3f} (일반 범위 밖이지만 그대로 사용)")

        # === 오른팔 초기값 (한 번만) ===
        if all(n in name_to_idx for n in RIGHT):
            self.robot_current['right'] = [msg.position[name_to_idx[n]] for n in RIGHT]

            if self.robot_initial['right'] is None:
                self.robot_initial['right'] = self.robot_current['right'][:]
                print(f"✅ 오른팔 하드웨어 초기값: {self._fmt(self.robot_initial['right'])}")

                # 초기값 분석 (정보 제공용)
                for i, val in enumerate(self.robot_initial['right']):
                    if val > 1.5 or val < -1.5:
                        print(f"  ℹ️  Joint{i+1}: {val:.3f} (일반 범위 밖이지만 그대로 사용)")

    def setup_socket(self):
        """소켓 연결 설정"""
        def socket_receiver():
            while True:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('localhost', 12345))
                    sock.settimeout(0.1)
                    print("🔗 VR 브릿지 연결 성공!")

                    buffer = ""
                    first_packet = True

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
                                        packet = json.loads(line)

                                        # 왼팔 데이터
                                        if 'left_arm' in packet and 'joint_angles' in packet['left_arm']:
                                            self.mujoco_current['left'] = packet['left_arm']['joint_angles'][:4]

                                            if first_packet and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'][:]
                                                print(f"✅ 왼팔 MuJoCo 초기값: {self._fmt(self.mujoco_initial['left'])}")

                                        # 오른팔 데이터
                                        if 'right_arm' in packet and 'joint_angles' in packet['right_arm']:
                                            self.mujoco_current['right'] = packet['right_arm']['joint_angles'][:4]

                                            if first_packet and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'][:]
                                                print(f"✅ 오른팔 MuJoCo 초기값: {self._fmt(self.mujoco_initial['right'])}")
                                                first_packet = False

                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    print(f"⚠️  소켓 연결 실패: {e}")
                    time.sleep(2)

        threading.Thread(target=socket_receiver, daemon=True).start()

    def create_joint_trajectory(self, target_joints, arm_side='left'):
        """조인트 궤적 메시지 생성"""
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

    def dual_arm_control(self):
        """양팔 제어 - 단일팔처럼 순수 오프셋만 적용"""

        # === 왼팔 제어 ===
        if (self.robot_initial['left'] is not None and
            self.mujoco_initial['left'] is not None):

            # 변화량 계산 (단일팔 mirror2.py와 동일)
            left_target = []
            for i in range(4):
                delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                target_val = self.robot_initial['left'][i] + delta
                left_target.append(target_val)

            # 첫 제어 시 디버그 출력
            if self._first_control['left']:
                print("\n" + "="*60)
                print("🎯 왼팔 첫 제어 (리밋 없음)")
                print(f"  하드웨어 초기: {self._fmt(self.robot_initial['left'])}")
                print(f"  MuJoCo 초기: {self._fmt(self.mujoco_initial['left'])}")
                print(f"  MuJoCo 현재: {self._fmt(self.mujoco_current['left'])}")
                delta_values = []
                for i in range(4):
                    delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                    delta_values.append(delta)
                print(f"  변화량: {self._fmt(delta_values)}")
                print(f"  목표값: {self._fmt(left_target)}")
                print("  💡 리밋 적용 없음 - MuJoCo에서만 체크")
                print("="*60)
                self._first_control['left'] = False

            # 궤적 생성 및 전송 (리밋 없이)
            left_traj = self.create_joint_trajectory(left_target, 'left')
            self.left_joint_pub.publish(left_traj)

        # === 오른팔 제어 ===
        if (self.robot_initial['right'] is not None and
            self.mujoco_initial['right'] is not None):

            # 변화량 계산 (리밋 없이)
            right_target = []
            for i in range(4):
                delta = self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                target_val = self.robot_initial['right'][i] + delta
                right_target.append(target_val)

            # 첫 제어 시 디버그 출력
            if self._first_control['right']:
                print("\n" + "="*60)
                print("🎯 오른팔 첫 제어 (리밋 없음)")
                print(f"  하드웨어 초기: {self._fmt(self.robot_initial['right'])}")
                print(f"  MuJoCo 초기: {self._fmt(self.mujoco_initial['right'])}")
                print(f"  MuJoCo 현재: {self._fmt(self.mujoco_current['right'])}")
                delta_values = []
                for i in range(4):
                    delta = self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                    delta_values.append(delta)
                print(f"  변화량: {self._fmt(delta_values)}")
                print(f"  목표값: {self._fmt(right_target)}")
                print("  💡 리밋 적용 없음 - MuJoCo에서만 체크")
                print("="*60)
                self._first_control['right'] = False

            # 궤적 생성 및 전송 (리밋 없이)
            right_traj = self.create_joint_trajectory(right_target, 'right')
            self.right_joint_pub.publish(right_traj)

    def print_status(self):
        """상태 출력"""
        print(f"\n📊 시스템 상태 [{time.strftime('%H:%M:%S')}]")

        # 초기값 수집 상태
        status_left = "✅" if self.robot_initial['left'] and self.mujoco_initial['left'] else "⏳"
        status_right = "✅" if self.robot_initial['right'] and self.mujoco_initial['right'] else "⏳"
        print(f"  왼팔: {status_left}  |  오른팔: {status_right}")

        # 현재 델타
        if self.mujoco_initial['left'] and self.mujoco_current['left']:
            left_delta = [
                self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                for i in range(4)
            ]
            # 의미있는 델타만 표시
            if any(abs(d) > 0.01 for d in left_delta):
                print(f"  왼팔 Δ: {self._fmt(left_delta, nd=3)}")

        if self.mujoco_initial['right'] and self.mujoco_current['right']:
            right_delta = [
                self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                for i in range(4)
            ]
            # 의미있는 델타만 표시
            if any(abs(d) > 0.01 for d in right_delta):
                print(f"  오른팔 Δ: {self._fmt(right_delta, nd=3)}")

    def _fmt(self, arr, nd=4):
        """배열 포맷팅 헬퍼"""
        try:
            return "[" + ", ".join(f"{x:+.{nd}f}" for x in arr) + "]"
        except:
            return str(arr)

def main():
    rclpy.init()

    try:
        node = DualArmNoLimitsMirror()

        print("\n" + "="*70)
        print("🚀 DUAL ARM NO LIMITS MIRROR 실행 중")
        print("="*70)
        print("✨ 주요 특징:")
        print("  • 단일팔처럼 리밋 없이 순수 오프셋만 적용")
        print("  • 초기 자세 그대로 유지 (점프 없음)")
        print("  • MuJoCo에서만 범위 체크 (이중 제한 방지)")
        print("  • 부드럽고 자연스러운 움직임")
        print("\n🎮 동작 방식:")
        print("  target = robot_initial + (mujoco_current - mujoco_initial)")
        print("  리밋 체크 없음 → MuJoCo가 알아서 처리")
        print("\n📋 실행 순서:")
        print("1. 하드웨어 런치 파일 실행")
        print("2. VR 브릿지 실행 (도커)")
        print("3. MuJoCo 시뮬레이션 실행")
        print("4. VR 컨트롤러 A+B 캘리브레이션")
        print("="*70 + "\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n🏁 종료")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()