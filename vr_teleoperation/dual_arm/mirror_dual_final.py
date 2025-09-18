#!/usr/bin/env python3
"""
미러 코드 최종 버전 - 델타 데드밴드 적용
델타가 거의 0일 때는 클리핑/속도제한 없이 현재 자세 유지
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import socket, json, threading, time
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand

class DualArmMirrorFinal(Node):
    def __init__(self):
        super().__init__('dual_arm_mirror_final')

        print("\n🤖 === 양팔 미러링 최종 버전 ===")
        print("✅ 델타 데드밴드 적용: 변화 없으면 움직이지 않음")

        # 초기값 저장
        self.robot_initial = {'left': None, 'right': None}
        self.mujoco_initial = {'left': None, 'right': None}
        self.mujoco_current = {'left': [0.0]*4, 'right': [0.0]*4}
        self.gripper_values = {'left': -0.01, 'right': -0.01}
        self.last_gripper_values = {'left': -0.01, 'right': -0.01}

        # 델타 데드밴드 (이 값보다 작으면 클리핑/속도제한 생략)
        self.delta_deadband = 0.02  # rad

        # 로봇 상태 추적
        self.robot_status = {
            'left_connected': False,
            'right_connected': False,
            'left_control_count': 0,
            'right_control_count': 0
        }

        # 안전 제한
        self.joint_limits = [
            [-3.14, 3.14],   # J1
            [-1.5, 1.5],     # J2
            [-1.5, 1.4],     # J3
            [-1.7, 1.97]     # J4
        ]

        # Publishers
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Gripper Action Clients
        self.left_gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(
            self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)

        # Socket
        self.setup_socket()

        # Control timer
        self.timer = self.create_timer(0.05, self.dual_arm_control)  # 20Hz

        # Status timer
        self.status_timer = self.create_timer(3.0, self.print_status)

        print("📡 초기값 수집 중...")

    def joint_states_callback(self, msg):
        """초기값 저장 - 이름 기반 동적 매핑"""
        # 왼쪽 로봇
        if self.robot_initial['left'] is None:
            left_idx = {}
            for i, name in enumerate(msg.name):
                if name in ('joint1', 'joint2', 'joint3', 'joint4'):
                    left_idx[name] = i

            if len(left_idx) == 4:
                order = ['joint1', 'joint2', 'joint3', 'joint4']
                self.robot_initial['left'] = [msg.position[left_idx[n]] for n in order]

                # 초기값 그대로 사용 (클리핑 없음)
                print(f"✅ 왼쪽 로봇 초기값: {[f'{x:.3f}' for x in self.robot_initial['left']]}")

                # 범위 초과 경고만
                for i, val in enumerate(self.robot_initial['left']):
                    if val < self.joint_limits[i][0] or val > self.joint_limits[i][1]:
                        print(f"   ⚠️ 왼팔 Joint{i+1}: {val:.3f} (범위 밖이지만 유지)")

                self.robot_status['left_connected'] = True
                # 첫 스텝 속도 제한용
                self.last_left_joints = self.robot_initial['left'].copy()

        # 오른쪽 로봇
        if self.robot_initial['right'] is None:
            right_idx = {}
            for i, name in enumerate(msg.name):
                if name in ('right_joint1', 'right_joint2', 'right_joint3', 'right_joint4'):
                    right_idx[name] = i

            if len(right_idx) == 4:
                order = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
                self.robot_initial['right'] = [msg.position[right_idx[n]] for n in order]

                print(f"✅ 오른쪽 로봇 초기값: {[f'{x:.3f}' for x in self.robot_initial['right']]}")

                # 범위 초과 경고만
                for i, val in enumerate(self.robot_initial['right']):
                    if val < self.joint_limits[i][0] or val > self.joint_limits[i][1]:
                        print(f"   ⚠️ 오른팔 Joint{i+1}: {val:.3f} (범위 밖이지만 유지)")

                self.robot_status['right_connected'] = True
                self.last_right_joints = self.robot_initial['right'].copy()

    def setup_socket(self):
        """MuJoCo 소켓 연결"""
        def recv():
            while True:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('localhost', 12345))
                    sock.settimeout(0.1)
                    print("🔗 MuJoCo 양팔 브릿지 연결됨")

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

                                        # 왼쪽 팔 데이터
                                        if 'left_arm' in d and 'joint_angles' in d['left_arm']:
                                            self.mujoco_current['left'] = d['left_arm']['joint_angles'][:4]

                                            if first and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'].copy()
                                                print(f"✅ MuJoCo 왼쪽 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")

                                        if 'left_arm' in d and 'gripper' in d['left_arm']:
                                            self.gripper_values['left'] = d['left_arm']['gripper']

                                        # 오른쪽 팔 데이터
                                        if 'right_arm' in d and 'joint_angles' in d['right_arm']:
                                            self.mujoco_current['right'] = d['right_arm']['joint_angles'][:4]

                                            if first and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'].copy()
                                                print(f"✅ MuJoCo 오른쪽 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['right']]}")
                                                first = False

                                        if 'right_arm' in d and 'gripper' in d['right_arm']:
                                            self.gripper_values['right'] = d['right_arm']['gripper']

                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    print(f"⚠️ MuJoCo 연결 오류: {e}")
                    time.sleep(2)

        threading.Thread(target=recv, daemon=True).start()

    def apply_safety_limits(self, joints, arm_side='left'):
        """안전 제한 적용 (속도 제한 + 범위 클리핑)"""
        safe = []
        for i, (val, lim) in enumerate(zip(joints, self.joint_limits)):
            # 속도 제한 0.1rad/step
            max_change = 0.1
            last_attr = f'last_{arm_side}_joints'
            if hasattr(self, last_attr):
                last = getattr(self, last_attr)
                if i < len(last):
                    dv = val - last[i]
                    if abs(dv) > max_change:
                        val = last[i] + np.sign(dv) * max_change

            # 범위 클리핑
            safe.append(np.clip(val, lim[0], lim[1]))

        setattr(self, f'last_{arm_side}_joints', safe.copy())
        return safe

    def create_joint_trajectory(self, target, arm_side='left'):
        """조인트 궤적 메시지 생성"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()

        if arm_side == 'right':
            traj.joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
        else:
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms

        traj.points = [pt]
        return traj

    def send_gripper_goal(self, position, arm_side='left'):
        """그리퍼 액션 전송"""
        client = self.left_gripper_client if arm_side == 'left' else self.right_gripper_client

        if not client.wait_for_server(timeout_sec=0.5):
            print(f"⚠️ {arm_side.upper()} 그리퍼 서버 연결 실패")
            return

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 100.0

        client.send_goal_async(goal)

    def dual_arm_control(self):
        """양팔 오프셋 기반 제어 - 델타 데드밴드 적용"""
        # 왼쪽 팔
        if self.robot_initial['left'] is not None and self.mujoco_initial['left'] is not None:
            # 변화량 계산
            deltas = [
                self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                for i in range(4)
            ]

            # 🔴 델타가 거의 0이면: 클리핑/속도제한 없이 현재 자세 유지
            if max(abs(d) for d in deltas) < self.delta_deadband:
                # 현재 자세 그대로 유지 (초기값 그대로 재전송)
                self.left_joint_pub.publish(
                    self.create_joint_trajectory(self.robot_initial['left'], 'left')
                )
            else:
                # 델타가 있을 때만 정상 제어
                left_target = [
                    self.robot_initial['left'][i] + deltas[i]
                    for i in range(4)
                ]

                # 안전 제한 적용
                safe_left = self.apply_safety_limits(left_target, 'left')

                # 전송
                self.left_joint_pub.publish(
                    self.create_joint_trajectory(safe_left, 'left')
                )

            # 그리퍼 제어
            if abs(self.gripper_values['left'] - self.last_gripper_values['left']) > 0.002:
                self.send_gripper_goal(self.gripper_values['left'], 'left')
                self.last_gripper_values['left'] = self.gripper_values['left']

            self.robot_status['left_control_count'] += 1

        # 오른쪽 팔
        if self.robot_initial['right'] is not None and self.mujoco_initial['right'] is not None:
            # 변화량 계산
            deltas = [
                self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                for i in range(4)
            ]

            # 🔴 델타가 거의 0이면: 클리핑/속도제한 없이 현재 자세 유지
            if max(abs(d) for d in deltas) < self.delta_deadband:
                # 현재 자세 그대로 유지
                self.right_joint_pub.publish(
                    self.create_joint_trajectory(self.robot_initial['right'], 'right')
                )
            else:
                # 델타가 있을 때만 정상 제어
                right_target = [
                    self.robot_initial['right'][i] + deltas[i]
                    for i in range(4)
                ]

                # 안전 제한 적용
                safe_right = self.apply_safety_limits(right_target, 'right')

                # 전송
                self.right_joint_pub.publish(
                    self.create_joint_trajectory(safe_right, 'right')
                )

            # 그리퍼 제어
            if abs(self.gripper_values['right'] - self.last_gripper_values['right']) > 0.002:
                self.send_gripper_goal(self.gripper_values['right'], 'right')
                self.last_gripper_values['right'] = self.gripper_values['right']

            self.robot_status['right_control_count'] += 1

    def print_status(self):
        """상태 정보 출력"""
        print(f"\n🤖 === 양팔 미러링 상태 ===")
        print(f"🔗 연결: 왼쪽={'✅' if self.robot_status['left_connected'] else '❌'} "
              f"오른쪽={'✅' if self.robot_status['right_connected'] else '❌'}")

        left_ready = (self.robot_initial['left'] is not None and
                     self.mujoco_initial['left'] is not None)
        right_ready = (self.robot_initial['right'] is not None and
                      self.mujoco_initial['right'] is not None)

        print(f"🎯 제어 준비: 왼쪽={'✅' if left_ready else '❌'} "
              f"오른쪽={'✅' if right_ready else '❌'}")
        print(f"📊 제어 횟수: 왼쪽={self.robot_status['left_control_count']} "
              f"오른쪽={self.robot_status['right_control_count']}")

        # 그리퍼 상태
        print(f"🖐 그리퍼: 왼쪽={self.gripper_values['left']:.3f} "
              f"오른쪽={self.gripper_values['right']:.3f}")

        # 델타 체크
        if left_ready:
            left_deltas = [
                self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                for i in range(4)
            ]
            max_delta = max(abs(d) for d in left_deltas)
            if max_delta < self.delta_deadband:
                print(f"🧊 왼팔: 델타 < {self.delta_deadband:.3f} → 자세 유지 중")
            else:
                print(f"🔄 왼팔: 활성 제어 중 (최대 델타: {max_delta:.3f})")

        if right_ready:
            right_deltas = [
                self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                for i in range(4)
            ]
            max_delta = max(abs(d) for d in right_deltas)
            if max_delta < self.delta_deadband:
                print(f"🧊 오른팔: 델타 < {self.delta_deadband:.3f} → 자세 유지 중")
            else:
                print(f"🔄 오른팔: 활성 제어 중 (최대 델타: {max_delta:.3f})")

    def emergency_stop(self):
        """비상 정지"""
        print("🛑 비상 정지 실행!")

        # 현재 위치 유지
        if self.robot_initial['left'] is not None:
            stop_traj = self.create_joint_trajectory(self.robot_initial['left'], 'left')
            self.left_joint_pub.publish(stop_traj)

        if self.robot_initial['right'] is not None:
            stop_traj = self.create_joint_trajectory(self.robot_initial['right'], 'right')
            self.right_joint_pub.publish(stop_traj)

        # 그리퍼 유지
        self.send_gripper_goal(self.last_gripper_values['left'], 'left')
        self.send_gripper_goal(self.last_gripper_values['right'], 'right')

def main():
    rclpy.init()

    try:
        dual_mirror = DualArmMirrorFinal()

        print("\n🤖 === 양팔 오프셋 미러링 시스템 (최종) ===")
        print("✅ 델타 데드밴드: 변화 없으면 움직이지 않음")
        print("✅ 초기 범위 초과 허용: 움직일 때만 안전 제한")
        print("🎯 왼쪽 VR → 왼쪽 로봇")
        print("🎯 오른쪽 VR → 오른쪽 로봇")
        print("🛑 Ctrl+C: 비상 정지")

        rclpy.spin(dual_mirror)

    except KeyboardInterrupt:
        print("\n🛑 비상 정지 요청됨")
        if 'dual_mirror' in locals():
            dual_mirror.emergency_stop()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        print("🏁 시스템 종료")

if __name__ == '__main__':
    main()