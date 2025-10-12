#!/usr/bin/env python3
"""
🤖 Dual-Arm VR → Physical Robot Bridge (Safe Version)
Per-step Headroom Limiter 적용으로 첫 프레임 점프 방지
- 리밋은 URDF 기준 유지
- 증분 제한으로 부드러운 시작
- 초기값 시드 자동 설정
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import socket
import json
import numpy as np
import threading
import time

class DualArmSafeMirror(Node):
    def __init__(self):
        super().__init__('dual_arm_safe_mirror')

        print("\n" + "="*70)
        print("🤖 DUAL ARM SAFE MIRROR")
        print("Per-step Headroom Limiter로 첫 프레임 점프 방지")
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
        # self.mujoco_current 아래에 추가
        self.gripper_values = {'left': -0.01, 'right': -0.01}
        self.last_gripper_values = {'left': -0.01, 'right': -0.01}

        # === 중요: 이전 명령값 저장 (Per-step용) ===
        self.last_left_joints = None
        self.last_right_joints = None

        # === Joint limits (URDF 기준 유지) ===
        self.joint_limits = [
            [-3.14, 3.14],   # Joint 1
            [-1.5, 1.5],     # Joint 2
            [-1.5, 1.4],     # Joint 3 (리밋 그대로!)
            [-1.7, 1.97]     # Joint 4
        ]

        # === 안전 파라미터 ===
        self.MAX_CHANGE_PER_STEP = 0.1  # rad/step (20Hz에서 2 rad/s)
        self.WARMUP_FRAMES = 5  # 처음 5프레임은 더 천천히
        self.warmup_counter = 0

        # === 진단 플래그 ===
        self._first_control = {
            'left': True,
            'right': True
        }
        self._name_vec = None  # joint_states 순서 변화 감지용

        # === Publishers ===
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # === Subscriber ===
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        
        # 그리퍼 Action Clients
        self.left_gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

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
            print("\n⚠️  /joint_states 순서 변경! 재매핑...")
            self.warmup_counter = 0  # 웜업 재시작

        # 이름 → 인덱스 매핑
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # 왼팔 조인트
        LEFT = ['joint1', 'joint2', 'joint3', 'joint4']
        # 오른팔 조인트
        RIGHT = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']

        # === 왼팔 초기값 ===
        if all(n in name_to_idx for n in LEFT):
            self.robot_current['left'] = [msg.position[name_to_idx[n]] for n in LEFT]

            if self.robot_initial['left'] is None:
                self.robot_initial['left'] = self.robot_current['left'][:]

                # 🔴 중요: 즉시 시드 설정!
                self.last_left_joints = self.robot_initial['left'][:]

                print(f"✅ 왼팔 초기값: {self._fmt(self.robot_initial['left'])}")

                # 리밋 체크
                for i, (val, (lo, hi)) in enumerate(zip(self.robot_initial['left'], self.joint_limits)):
                    if val < lo or val > hi:
                        print(f"  ⚠️  Joint{i+1} 리밋 초과: {val:.3f} (범위: [{lo:.2f}, {hi:.2f}])")

        # === 오른팔 초기값 ===
        if all(n in name_to_idx for n in RIGHT):
            self.robot_current['right'] = [msg.position[name_to_idx[n]] for n in RIGHT]

            if self.robot_initial['right'] is None:
                self.robot_initial['right'] = self.robot_current['right'][:]

                # 🔴 중요: 즉시 시드 설정!
                self.last_right_joints = self.robot_initial['right'][:]

                print(f"✅ 오른팔 초기값: {self._fmt(self.robot_initial['right'])}")

                # 리밋 체크
                for i, (val, (lo, hi)) in enumerate(zip(self.robot_initial['right'], self.joint_limits)):
                    if val < lo or val > hi:
                        print(f"  ⚠️  Joint{i+1} 리밋 초과: {val:.3f} (범위: [{lo:.2f}, {hi:.2f}])")

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
                                            # ========= [3] 여기에 그리퍼 값 수신 코드 추가 =========
                                            if 'gripper' in packet['left_arm']:
                                                self.gripper_values['left'] = packet['left_arm']['gripper']
                                            # =================================================

                                            if first_packet and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'][:]
                                                print(f"✅ 왼팔 MuJoCo 초기값: {self._fmt(self.mujoco_initial['left'])}")

                                        # 오른팔 데이터
                                        if 'right_arm' in packet and 'joint_angles' in packet['right_arm']:
                                            self.mujoco_current['right'] = packet['right_arm']['joint_angles'][:4]
                                            # ========= [3] 여기에 그리퍼 값 수신 코드 추가 =========
                                            if 'gripper' in packet['right_arm']:
                                                self.gripper_values['right'] = packet['right_arm']['gripper']
                                            # =================================================

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

    def apply_per_step_limiter(self, target_joints, arm_side='left'):
        """
        Per-step Headroom Limiter
        - 절대값 클램프 대신 증분만 제한
        - 첫 프레임 점프 방지
        """
        # 이전 명령값 가져오기
        last = self.last_left_joints if arm_side == 'left' else self.last_right_joints

        # 첫 호출 시 시드 확인
        if last is None:
            last = self.robot_initial[arm_side][:]
            print(f"⚠️  {arm_side} 시드 초기화: {self._fmt(last)}")

        safe_joints = []

        # 웜업 중에는 더 작은 변화율
        if self.warmup_counter < self.WARMUP_FRAMES:
            max_change = 0.03  # 웜업: 0.6 rad/s
        else:
            max_change = self.MAX_CHANGE_PER_STEP  # 정상: 2 rad/s

        for i, (desired, prev) in enumerate(zip(target_joints, last)):
            lo, hi = self.joint_limits[i]

            # 증분 계산
            step = desired - prev

            # 헤드룸 제한 (리밋 방향으로만)
            # 현재 위치에서 갈 수 있는 최대 범위
            if prev + step > hi:
                step = min(step, hi - prev)  # 상한까지만
            elif prev + step < lo:
                step = max(step, lo - prev)  # 하한까지만

            # 변화율 제한
            step = np.clip(step, -max_change, max_change)

            # 최종 목표
            new_val = prev + step
            safe_joints.append(new_val)

            # 첫 제어 시 로그
            if self._first_control[arm_side] and abs(step) > 0.001:
                if abs(desired - new_val) > 0.01:
                    print(f"  [{arm_side}] Joint{i+1}: {desired:.3f} → {new_val:.3f} (제한됨)")

        # 현재 명령값 업데이트 (다음 프레임 기준)
        if arm_side == 'left':
            self.last_left_joints = safe_joints[:]
        else:
            self.last_right_joints = safe_joints[:]

        return safe_joints

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
    
    def send_gripper_goal(self, position, arm_side='left'):
        """그리퍼 액션 전송"""
        client = self.left_gripper_client if arm_side == 'left' else self.right_gripper_client
        last_value = self.last_gripper_values[arm_side]

        # 값이 거의 변하지 않았으면 보내지 않음
        if abs(position - last_value) < 0.002:
            return

        if not client.wait_for_server(timeout_sec=0.05):
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)
        client.send_goal_async(goal_msg)
        self.last_gripper_values[arm_side] = position
        
    def dual_arm_control(self):
        """양팔 제어 - Per-step Limiter 적용"""
        # 웜업 카운터
        if self.warmup_counter < self.WARMUP_FRAMES:
            self.warmup_counter += 1
            if self.warmup_counter == 1:
                print(f"🔄 웜업 시작 ({self.WARMUP_FRAMES} 프레임)")
            elif self.warmup_counter == self.WARMUP_FRAMES:
                print("✅ 웜업 완료")

        # === 왼팔 제어 ===
        if (self.robot_initial['left'] is not None and
            self.mujoco_initial['left'] is not None):

            # 델타 계산
            left_delta = [
                self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                for i in range(4)
            ]

            # 목표값 계산 (클리핑 전)
            left_target = [
                self.robot_initial['left'][i] + left_delta[i]
                for i in range(4)
            ]

            # 첫 제어 프레임 로그
            if self._first_control['left']:
                print("\n" + "="*60)
                print("🎯 왼팔 첫 제어 프레임")
                print(f"  초기값: {self._fmt(self.robot_initial['left'])}")
                print(f"  델타:   {self._fmt(left_delta)}")
                print(f"  목표값: {self._fmt(left_target)}")
                self._first_control['left'] = False

            # Per-step Limiter 적용
            safe_left = self.apply_per_step_limiter(left_target, 'left')

            # 궤적 생성 및 발행
            left_traj = self.create_joint_trajectory(safe_left, 'left')
            self.left_joint_pub.publish(left_traj)
            self.send_gripper_goal(self.gripper_values['left'], 'left')

        # === 오른팔 제어 ===
        if (self.robot_initial['right'] is not None and
            self.mujoco_initial['right'] is not None):

            # 델타 계산
            right_delta = [
                self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                for i in range(4)
            ]

            # 목표값 계산 (클리핑 전)
            right_target = [
                self.robot_initial['right'][i] + right_delta[i]
                for i in range(4)
            ]

            # 첫 제어 프레임 로그
            if self._first_control['right']:
                print("\n" + "="*60)
                print("🎯 오른팔 첫 제어 프레임")
                print(f"  초기값: {self._fmt(self.robot_initial['right'])}")
                print(f"  델타:   {self._fmt(right_delta)}")
                print(f"  목표값: {self._fmt(right_target)}")
                self._first_control['right'] = False

            # Per-step Limiter 적용
            safe_right = self.apply_per_step_limiter(right_target, 'right')

            # 궤적 생성 및 발행
            right_traj = self.create_joint_trajectory(safe_right, 'right')
            self.right_joint_pub.publish(right_traj)
            self.send_gripper_goal(self.gripper_values['right'], 'right')


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
            print(f"  왼팔 Δ: {self._fmt(left_delta, nd=3)}")

        if self.mujoco_initial['right'] and self.mujoco_current['right']:
            right_delta = [
                self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                for i in range(4)
            ]
            print(f"  오른팔 Δ: {self._fmt(right_delta, nd=3)}")

        # 웜업 상태
        if self.warmup_counter < self.WARMUP_FRAMES:
            print(f"  웜업: {self.warmup_counter}/{self.WARMUP_FRAMES}")

    def _fmt(self, arr, nd=4):
        """배열 포맷팅 헬퍼"""
        try:
            return "[" + ", ".join(f"{x:+.{nd}f}" for x in arr) + "]"
        except:
            return str(arr)

def main():
    rclpy.init()

    try:
        node = DualArmSafeMirror()

        print("\n" + "="*70)
        print("🚀 DUAL ARM SAFE MIRROR 실행 중")
        print("="*70)
        print("• Per-step Headroom Limiter로 첫 프레임 점프 방지")
        print("• 리밋은 URDF 기준 유지 (J3: 1.4, J4: -1.7)")
        print("• 초기 5프레임 웜업 모드")
        print("\n실행 순서:")
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
