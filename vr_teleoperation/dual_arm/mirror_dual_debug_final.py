#!/usr/bin/env python3
"""
미러 코드 디버깅 - 원본 mirror_dual.py와 완전히 동일 동작 + 왼팔 첫 제어 시점 로그 추가
+ 첫 스텝 속도 제한 추가
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

class DebugMirror(Node):
    def __init__(self):
        super().__init__('debug_mirror')

        print("\n🔍 === 미러 디버깅 모드 (원본과 동일 동작 + 첫 스텝 보호) ===")

        # 초기값 저장 (원본과 동일 구조)
        self.robot_initial = {'left': None, 'right': None}
        self.mujoco_initial = {'left': None, 'right': None}
        self.mujoco_current = {'left': [0.0]*4, 'right': [0.0]*4}
        self.gripper_values = {'left': -0.01, 'right': -0.01}
        self.last_gripper_values = {'left': -0.01, 'right': -0.01}

        # 첫 제어시점 로그 플래그
        self.first_control = {'left': True, 'right': True}

        # 안전 제한(원본과 동일)
        self.joint_limits = [
            [-3.14, 3.14],   # J1
            [-1.5, 1.5],     # J2
            [-1.5, 1.4],     # J3
            [-1.7, 1.97]     # J4
        ]

        # Publishers (원본 토픽과 동일)
        self.left_joint_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(JointTrajectory, '/right_arm_controller/joint_trajectory', 10)

        # Gripper Action Clients (원본과 동일)
        self.left_gripper_client  = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # Subscriber (이름 기반 동적 매핑: 원본과 동일)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)

        # 소켓 수신(원본과 동일)
        self.setup_socket()

        # 제어 타이머(원본과 동일 20Hz)
        self.timer = self.create_timer(0.05, self.debug_control)

        print("📡 초기값 수집 중... (/joint_states, 이름 기반 인덱싱)")

    def joint_states_callback(self, msg: JointState):
        """초기값 저장 - 이름 기반 동적 매핑 (원본과 동일)"""
        # 왼쪽 로봇
        if self.robot_initial['left'] is None:
            left_idx = {}
            for i, name in enumerate(msg.name):
                if name in ('joint1', 'joint2', 'joint3', 'joint4'):
                    left_idx[name] = i
            if len(left_idx) == 4:
                order = ['joint1','joint2','joint3','joint4']
                self.robot_initial['left'] = [msg.position[left_idx[n]] for n in order]
                print(f"✅ 왼팔 하드웨어 초기값: {[f'{x:.3f}' for x in self.robot_initial['left']]}")

                # 🔥 첫 스텝 속도 제한을 위해 last_joints를 초기값으로 설정
                self.last_left_joints = self.robot_initial['left'].copy()
                print(f"   → 첫 스텝 보호: last_left_joints 초기화됨")

        # 오른쪽 로봇
        if self.robot_initial['right'] is None:
            right_idx = {}
            for i, name in enumerate(msg.name):
                if name in ('right_joint1','right_joint2','right_joint3','right_joint4'):
                    right_idx[name] = i
            if len(right_idx) == 4:
                order = ['right_joint1','right_joint2','right_joint3','right_joint4']
                self.robot_initial['right'] = [msg.position[right_idx[n]] for n in order]
                print(f"✅ 오른팔 하드웨어 초기값: {[f'{x:.3f}' for x in self.robot_initial['right']]}")

                # 🔥 첫 스텝 속도 제한
                self.last_right_joints = self.robot_initial['right'].copy()

    def setup_socket(self):
        """MuJoCo 소켓 연결 (원본과 동일 로직)"""
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
                                if not line:
                                    continue
                                try:
                                    d = json.loads(line)

                                    # 왼팔
                                    if 'left_arm' in d:
                                        if 'joint_angles' in d['left_arm']:
                                            self.mujoco_current['left'] = d['left_arm']['joint_angles'][:4]
                                            if first and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'].copy()
                                                print(f"✅ 왼팔 MuJoCo 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")
                                        if 'gripper' in d['left_arm']:
                                            self.gripper_values['left'] = d['left_arm']['gripper']

                                    # 오른팔
                                    if 'right_arm' in d:
                                        if 'joint_angles' in d['right_arm']:
                                            self.mujoco_current['right'] = d['right_arm']['joint_angles'][:4]
                                            if first and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'].copy()
                                                print(f"✅ 오른팔 MuJoCo 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['right']]}")
                                                first = False
                                        if 'gripper' in d['right_arm']:
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
        """안전 제한 적용 (원본과 동일 + 첫 스텝도 보호)"""
        safe = []
        for i, (val, lim) in enumerate(zip(joints, self.joint_limits)):
            # 스텝당 변화 제한 0.1rad (원본과 동일)
            max_change = 0.1
            last_attr = f'last_{arm_side}_joints'
            if hasattr(self, last_attr):
                last = getattr(self, last_attr)
                if i < len(last):
                    dv = val - last[i]
                    if abs(dv) > max_change:
                        val = last[i] + np.sign(dv) * max_change
                        # 첫 제어에서 속도 제한 발생시 로그
                        if self.first_control.get(arm_side, False):
                            print(f"  🛡️ {arm_side} J{i+1} 속도제한: Δ={dv:.3f} → {np.sign(dv)*max_change:.3f}")
            safe.append(np.clip(val, lim[0], lim[1]))
        setattr(self, f'last_{arm_side}_joints', safe.copy())
        return safe

    def create_joint_trajectory(self, target, arm_side='left'):
        """조인트 궤적 메시지 (원본과 동일)"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        if arm_side == 'right':
            traj.joint_names = ['right_joint1','right_joint2','right_joint3','right_joint4']
        else:
            traj.joint_names = ['joint1','joint2','joint3','joint4']
        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms
        traj.points = [pt]
        return traj

    def send_gripper_goal(self, position, arm_side='left'):
        """그리퍼 액션 (원본과 동일)"""
        client = self.left_gripper_client if arm_side == 'left' else self.right_gripper_client
        if not client.wait_for_server(timeout_sec=0.5):
            print(f"⚠️ {arm_side.upper()} 그리퍼 서버 연결 실패")
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 100.0
        client.send_goal_async(goal)

    def debug_control(self):
        """제어 루프 (원본과 완전히 동일) + 왼팔 첫 제어시 로그만 추가"""
        # 왼팔
        if self.robot_initial['left'] is not None and self.mujoco_initial['left'] is not None:
            left_target = []
            for i in range(4):
                delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                left_target.append(self.robot_initial['left'][i] + delta)

            # 첫 제어 시점 분석 로그
            if self.first_control['left']:
                print("\n" + "="*60)
                print("🎯 왼팔 첫 제어 시점 분석")
                print(f"  HW 초기:    {[f'{x:.3f}' for x in self.robot_initial['left']]}")
                print(f"  MJ 초기:    {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")
                print(f"  MJ 현재:    {[f'{x:.3f}' for x in self.mujoco_current['left']]}")
                deltas = [self.mujoco_current['left'][i] - self.mujoco_initial['left'][i] for i in range(4)]
                print(f"  변화량 Δ:   {[f'{d:.3f}' for d in deltas]}")
                print(f"  목표(원본): {[f'{x:.3f}' for x in left_target]}")

                # 클리핑 체크
                clipped = [np.clip(v, self.joint_limits[i][0], self.joint_limits[i][1]) for i, v in enumerate(left_target)]
                for i in range(4):
                    if abs(left_target[i]-clipped[i]) > 1e-2:
                        print(f"  ⚠️ J{i+1} 범위클리핑: {left_target[i]:.3f} → {clipped[i]:.3f}")

                # 속도 제한 예측
                print(f"  이전값:     {[f'{x:.3f}' for x in self.last_left_joints]}")
                for i in range(4):
                    change = left_target[i] - self.last_left_joints[i]
                    if abs(change) > 0.1:
                        print(f"  🛡️ J{i+1} 속도제한 예상: Δ={change:.3f} → {np.sign(change)*0.1:.3f}")

                print("="*60)
                self.first_control['left'] = False

            safe_left = self.apply_safety_limits(left_target, 'left')
            self.left_joint_pub.publish(self.create_joint_trajectory(safe_left, 'left'))

            # 그리퍼 (원본과 동일 임계값)
            if abs(self.gripper_values['left'] - self.last_gripper_values['left']) > 0.002:
                self.send_gripper_goal(self.gripper_values['left'], 'left')
                self.last_gripper_values['left'] = self.gripper_values['left']

        # 오른팔 (원본과 동일, 추가 로그 없음)
        if self.robot_initial['right'] is not None and self.mujoco_initial['right'] is not None:
            right_target = []
            for i in range(4):
                delta = self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                right_target.append(self.robot_initial['right'][i] + delta)
            safe_right = self.apply_safety_limits(right_target, 'right')
            self.right_joint_pub.publish(self.create_joint_trajectory(safe_right, 'right'))

            if abs(self.gripper_values['right'] - self.last_gripper_values['right']) > 0.002:
                self.send_gripper_goal(self.gripper_values['right'], 'right')
                self.last_gripper_values['right'] = self.gripper_values['right']

def main():
    rclpy.init()
    try:
        node = DebugMirror()
        print("\n🔍 === 미러 디버깅 실행 중 (첫 스텝 보호 적용) ===")
        print("✅ 첫 제어에도 0.1 rad/step 속도 제한 적용")
        print("📊 왼팔 첫 제어 시점에서만 상세 로그 출력")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n디버깅 종료")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()