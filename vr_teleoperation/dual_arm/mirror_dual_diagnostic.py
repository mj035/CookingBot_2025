#!/usr/bin/env python3
"""
🔍 Dual Arm Mirror Diagnostic Version
왼팔 튀는 문제 진단을 위한 특별 버전
- 이름 기반 매핑
- 첫 프레임 상세 로깅
- 데드존 게이트
- 변화율 제한 시드
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

class MirrorDualDiagnostic(Node):
    def __init__(self):
        super().__init__('mirror_dual_diagnostic')

        print("\n" + "="*70)
        print("🔍 DUAL ARM MIRROR DIAGNOSTIC MODE")
        print("왼팔 튀는 문제 진단을 위한 특별 버전")
        print("="*70 + "\n")

        # === 진단 플래그 ===
        self._first_log_left_done = False
        self._first_log_right_done = False
        self._first_left_gate = True        # 데드존 게이트
        self._first_right_gate = True       # 데드존 게이트
        self._name_vec = None               # /joint_states name 배열 변동 감지용
        self._warmup_frames = 5             # 웜업 프레임 수
        self._warmup_counter = 0

        # 데드존 게이트용 카운터
        self._left_deadzone_counter = 0
        self._right_deadzone_counter = 0
        self._deadzone_threshold = 3  # 3프레임 연속 작은 델타

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

        # 이전 조인트값 (변화율 제한용)
        self.last_left_joints = None
        self.last_right_joints = None

        # === Joint limits ===
        self.joint_limits = [
            [-3.14, 3.14],   # Joint 1
            [-1.5, 1.5],     # Joint 2
            [-1.5, 1.4],     # Joint 3
            [-1.7, 1.97]     # Joint 4
        ]

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

    def _fmt(self, arr, nd=4):
        """배열 포맷팅 헬퍼"""
        try:
            return "[" + ", ".join(f"{x:+.{nd}f}" for x in arr) + "]"
        except:
            return str(arr)

    def _log_first_frame_left(self, *, left_initial, mjc_initial, mjc_current,
                              delta, target_preclip, target_postclip,
                              big_move_thresh=0.2):
        """왼팔 첫 프레임 진단 로그"""
        if self._first_log_left_done:
            return
        self._first_log_left_done = True

        print("\n" + "="*70)
        print("🧪 [LEFT ARM] 첫 프레임 진단 로그")
        print("="*70)
        print("  • 하드웨어 초기:      ", self._fmt(left_initial))
        print("  • MuJoCo 초기:        ", self._fmt(mjc_initial))
        print("  • MuJoCo 현재:        ", self._fmt(mjc_current))
        print("  • Δ(현재-초기):       ", self._fmt(delta))
        print("  • 목표(클리핑 전):    ", self._fmt(target_preclip))
        print("  • 목표(클리핑 후):    ", self._fmt(target_postclip))

        # 큰 움직임 경고
        for i, d in enumerate(delta):
            if abs(d) > big_move_thresh:
                print(f"  ⚠️  Joint{i+1} 큰 움직임 경고: Δ={d:+.4f} rad (>{big_move_thresh:.2f})")

        # 클리핑 발생 확인
        for i in range(4):
            if abs(target_preclip[i] - target_postclip[i]) > 0.001:
                print(f"  ⚠️  Joint{i+1} 클리핑 발생: {target_preclip[i]:+.4f} → {target_postclip[i]:+.4f}")

        print("="*70 + "\n")

    def _log_first_frame_right(self, *, right_initial, mjc_initial, mjc_current,
                               delta, target_preclip, target_postclip,
                               big_move_thresh=0.2):
        """오른팔 첫 프레임 진단 로그"""
        if self._first_log_right_done:
            return
        self._first_log_right_done = True

        print("\n" + "="*70)
        print("🧪 [RIGHT ARM] 첫 프레임 진단 로그")
        print("="*70)
        print("  • 하드웨어 초기:      ", self._fmt(right_initial))
        print("  • MuJoCo 초기:        ", self._fmt(mjc_initial))
        print("  • MuJoCo 현재:        ", self._fmt(mjc_current))
        print("  • Δ(현재-초기):       ", self._fmt(delta))
        print("  • 목표(클리핑 전):    ", self._fmt(target_preclip))
        print("  • 목표(클리핑 후):    ", self._fmt(target_postclip))

        # 큰 움직임 경고
        for i, d in enumerate(delta):
            if abs(d) > big_move_thresh:
                print(f"  ⚠️  Joint{i+1} 큰 움직임 경고: Δ={d:+.4f} rad (>{big_move_thresh:.2f})")

        # 클리핑 발생 확인
        for i in range(4):
            if abs(target_preclip[i] - target_postclip[i]) > 0.001:
                print(f"  ⚠️  Joint{i+1} 클리핑 발생: {target_preclip[i]:+.4f} → {target_postclip[i]:+.4f}")

        print("="*70 + "\n")

    def joint_states_callback(self, msg):
        """Joint States 콜백 - 이름 기반 매핑"""
        # 이름 순서 변화 감지
        if self._name_vec is None:
            self._name_vec = list(msg.name)
            print("\n📋 Joint States 구조 확인:")
            print(f"  총 조인트 수: {len(msg.name)}")
            print(f"  조인트 이름: {msg.name}")
        elif self._name_vec != list(msg.name):
            self._name_vec = list(msg.name)
            print("\n⚠️  /joint_states name 순서가 변경됨!")
            print(f"  새 순서: {msg.name}")
            self._warmup_counter = 0  # 웜업 재시작

        # 이름 → 인덱스 매핑 생성
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        # 왼팔 조인트 이름
        LEFT = ['joint1', 'joint2', 'joint3', 'joint4']
        # 오른팔 조인트 이름
        RIGHT = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']

        # 왼팔 - 필수 조인트 모두 존재할 때만 업데이트
        if all(n in name_to_idx for n in LEFT):
            self.robot_current['left'] = [msg.position[name_to_idx[n]] for n in LEFT]
            if self.robot_initial['left'] is None:
                self.robot_initial['left'] = self.robot_current['left'][:]
                print(f"✅ 왼팔 하드웨어 초기값 저장: {self._fmt(self.robot_initial['left'])}")
                # 첫 프레임 변화율 제한 시드
                self.last_left_joints = self.robot_initial['left'][:]
        else:
            missing = [n for n in LEFT if n not in name_to_idx]
            if missing and self.robot_initial['left'] is None:
                print(f"⚠️  왼팔 조인트 누락: {missing}")

        # 오른팔 - 필수 조인트 모두 존재할 때만 업데이트
        if all(n in name_to_idx for n in RIGHT):
            self.robot_current['right'] = [msg.position[name_to_idx[n]] for n in RIGHT]
            if self.robot_initial['right'] is None:
                self.robot_initial['right'] = self.robot_current['right'][:]
                print(f"✅ 오른팔 하드웨어 초기값 저장: {self._fmt(self.robot_initial['right'])}")
                # 첫 프레임 변화율 제한 시드
                self.last_right_joints = self.robot_initial['right'][:]
        else:
            missing = [n for n in RIGHT if n not in name_to_idx]
            if missing and self.robot_initial['right'] is None:
                print(f"⚠️  오른팔 조인트 누락: {missing}")

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
                                                print(f"✅ 왼팔 MuJoCo 초기값 저장: {self._fmt(self.mujoco_initial['left'])}")

                                        # 오른팔 데이터
                                        if 'right_arm' in packet and 'joint_angles' in packet['right_arm']:
                                            self.mujoco_current['right'] = packet['right_arm']['joint_angles'][:4]

                                            if first_packet and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'][:]
                                                print(f"✅ 오른팔 MuJoCo 초기값 저장: {self._fmt(self.mujoco_initial['right'])}")
                                                first_packet = False

                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    print(f"⚠️  소켓 연결 실패: {e}")
                    time.sleep(2)

        threading.Thread(target=socket_receiver, daemon=True).start()

    def apply_safety_limits(self, joints, arm_side='left'):
        """안전 제한 적용"""
        safe_joints = []
        last_joints = self.last_left_joints if arm_side == 'left' else self.last_right_joints

        for i, (joint_val, limits) in enumerate(zip(joints, self.joint_limits)):
            # 변화율 제한 (0.1 rad/step)
            max_change = 0.1
            if self._warmup_counter < self._warmup_frames:
                max_change = 0.03  # 웜업 중에는 더 작은 변화율

            if last_joints is not None and i < len(last_joints):
                change = joint_val - last_joints[i]
                if abs(change) > max_change:
                    joint_val = last_joints[i] + np.sign(change) * max_change

            # 조인트 범위 제한
            safe_joint = np.clip(joint_val, limits[0], limits[1])
            safe_joints.append(safe_joint)

        # 현재 조인트 저장
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

    def _small_delta(self, vec, tol=0.03):
        """델타가 충분히 작은지 확인"""
        return all(abs(x) < tol for x in vec)

    def dual_arm_control(self):
        """양팔 제어 - 진단 모드"""
        # 웜업 카운터
        if self._warmup_counter < self._warmup_frames:
            self._warmup_counter += 1
            if self._warmup_counter == 1:
                print(f"🔄 웜업 모드 시작 ({self._warmup_frames} 프레임)")
            elif self._warmup_counter == self._warmup_frames:
                print("✅ 웜업 완료")

        # === 왼팔 제어 ===
        if (self.robot_initial['left'] is not None and
            self.mujoco_initial['left'] is not None):

            # (1) Δ 계산
            mjc_left_initial = self.mujoco_initial['left']
            mjc_left_current = self.mujoco_current['left']
            left_initial = self.robot_initial['left']
            left_delta = [mjc_left_current[i] - mjc_left_initial[i] for i in range(4)]

            # (2) 데드존 게이트
            if self._first_left_gate:
                if self._small_delta(left_delta, 0.03):
                    self._left_deadzone_counter += 1
                    if self._left_deadzone_counter >= self._deadzone_threshold:
                        self._first_left_gate = False
                        print(f"✅ 왼팔 데드존 통과 (델타 안정화 확인)")
                else:
                    self._left_deadzone_counter = 0
                    if self._left_deadzone_counter == 0 and not self._first_log_left_done:
                        print(f"⏳ 왼팔 데드존 대기중... Δ={self._fmt(left_delta)}")
                    return  # 아직 제어 시작 안 함

            # (3) 목표값 계산 (클리핑 전)
            left_target_preclip = [left_initial[i] + left_delta[i] for i in range(4)]

            # (4) 안전 제한 적용
            left_target_postclip = self.apply_safety_limits(left_target_preclip, 'left')

            # (5) 첫 프레임 로그
            self._log_first_frame_left(
                left_initial=left_initial,
                mjc_initial=mjc_left_initial,
                mjc_current=mjc_left_current,
                delta=left_delta,
                target_preclip=left_target_preclip,
                target_postclip=left_target_postclip,
                big_move_thresh=0.2
            )

            # (6) 궤적 생성 및 발행
            left_traj = self.create_joint_trajectory(left_target_postclip, 'left')
            self.left_joint_pub.publish(left_traj)

        # === 오른팔 제어 ===
        if (self.robot_initial['right'] is not None and
            self.mujoco_initial['right'] is not None):

            # (1) Δ 계산
            mjc_right_initial = self.mujoco_initial['right']
            mjc_right_current = self.mujoco_current['right']
            right_initial = self.robot_initial['right']
            right_delta = [mjc_right_current[i] - mjc_right_initial[i] for i in range(4)]

            # (2) 데드존 게이트
            if self._first_right_gate:
                if self._small_delta(right_delta, 0.03):
                    self._right_deadzone_counter += 1
                    if self._right_deadzone_counter >= self._deadzone_threshold:
                        self._first_right_gate = False
                        print(f"✅ 오른팔 데드존 통과 (델타 안정화 확인)")
                else:
                    self._right_deadzone_counter = 0
                    if self._right_deadzone_counter == 0 and not self._first_log_right_done:
                        print(f"⏳ 오른팔 데드존 대기중... Δ={self._fmt(right_delta)}")
                    return  # 아직 제어 시작 안 함

            # (3) 목표값 계산 (클리핑 전)
            right_target_preclip = [right_initial[i] + right_delta[i] for i in range(4)]

            # (4) 안전 제한 적용
            right_target_postclip = self.apply_safety_limits(right_target_preclip, 'right')

            # (5) 첫 프레임 로그
            self._log_first_frame_right(
                right_initial=right_initial,
                mjc_initial=mjc_right_initial,
                mjc_current=mjc_right_current,
                delta=right_delta,
                target_preclip=right_target_preclip,
                target_postclip=right_target_postclip,
                big_move_thresh=0.2
            )

            # (6) 궤적 생성 및 발행
            right_traj = self.create_joint_trajectory(right_target_postclip, 'right')
            self.right_joint_pub.publish(right_traj)

    def print_status(self):
        """상태 출력"""
        print(f"\n📊 시스템 상태 [{time.strftime('%H:%M:%S')}]")

        # 초기값 수집 상태
        status_left = "✅" if self.robot_initial['left'] and self.mujoco_initial['left'] else "⏳"
        status_right = "✅" if self.robot_initial['right'] and self.mujoco_initial['right'] else "⏳"
        print(f"  왼팔: {status_left}  |  오른팔: {status_right}")

        # 데드존 상태
        if self._first_left_gate:
            print(f"  왼팔 데드존: 대기중 ({self._left_deadzone_counter}/{self._deadzone_threshold})")
        if self._first_right_gate:
            print(f"  오른팔 데드존: 대기중 ({self._right_deadzone_counter}/{self._deadzone_threshold})")

        # 현재 델타
        if self.mujoco_initial['left'] and self.mujoco_current['left']:
            left_delta = [self.mujoco_current['left'][i] - self.mujoco_initial['left'][i] for i in range(4)]
            print(f"  왼팔 Δ: {self._fmt(left_delta, nd=3)}")

        if self.mujoco_initial['right'] and self.mujoco_current['right']:
            right_delta = [self.mujoco_current['right'][i] - self.mujoco_initial['right'][i] for i in range(4)]
            print(f"  오른팔 Δ: {self._fmt(right_delta, nd=3)}")

def main():
    rclpy.init()

    try:
        node = MirrorDualDiagnostic()

        print("\n" + "="*70)
        print("🔍 진단 모드 실행 지시사항")
        print("="*70)
        print("1. 하드웨어 런치 파일 실행 (초기 자세 설정)")
        print("2. VR 브릿지 실행 (도커 컨테이너)")
        print("3. MuJoCo 시뮬레이션 실행")
        print("4. VR 컨트롤러 A+B 버튼으로 캘리브레이션")
        print("5. 왼팔 움직여서 첫 제어 시작")
        print("\n첫 프레임 로그가 자동으로 출력됩니다.")
        print("="*70 + "\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n🏁 진단 모드 종료")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()