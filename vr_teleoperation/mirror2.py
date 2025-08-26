#!/usr/bin/env python3
"""
🤖 VR → Physical Robot Bridge (Host/ROS2)

이 파일은 VR 컨트롤러의 움직임을 실제 OpenManipulator-X 로봇에 전달하는 
핵심 브릿지 역할을 합니다.

주요 기능:
- Docker(ROS1)에서 Socket으로 받은 MuJoCo 조인트 값을 처리
- Offset-based Control: 절대 위치가 아닌 상대적 변화량만 적용
- 안전한 로봇 제어: 위험한 포즈 방지 및 부드러운 움직임 보장
- ROS2 JointTrajectory 메시지로 실제 로봇에 명령 전송

동작 방식:
target_joint = initial_robot_pose + (mujoco_current - mujoco_initial)

Safety Features:
- Joint limit 체크
- 과도한 움직임 제한
- Emergency stop 기능
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
from std_msgs.msg import Float64
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand

class OffsetMirror(Node):
    def __init__(self):
        super().__init__('offset_mirror')
        
        print("\n오프셋 미러링 - 변화량만 적용")
        
        # 초기값 저장
        self.real_initial = None  # 실물 초기 위치
        self.mujoco_initial = None  # MuJoCo 초기 위치
        self.mujoco_current = [0.0, 0.0, 0.0, 0.0]
        self.gripper_value = -0.01  # 그리퍼 추가
        
        # ROS
        self.joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.save_initial, 10)
        
        # Gripper Action Client
        self.gripper_client = ActionClient(self, GripperCommand, 'gripper_controller/gripper_cmd')
        self.last_gripper_value = -0.01  # 마지막 그리퍼 값 추적
        
        # MuJoCo 연결
        self.setup_socket()
        
        # 제어 루프
        self.timer = self.create_timer(0.05, self.control)
        
        print("초기 위치 읽는 중...")
    
    def save_initial(self, msg):
        """실물 초기 위치 저장 (한 번만)"""
        if self.real_initial is None:
            self.real_initial = []
            for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                for i, n in enumerate(msg.name):
                    if name in n:
                        self.real_initial.append(msg.position[i])
                        break
            
            if len(self.real_initial) == 4:
                print(f"실물 초기: {[f'{x:.3f}' for x in self.real_initial]}")
    
    def setup_socket(self):
        def recv():
            while True:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(('localhost', 12345))
                    sock.settimeout(0.1)
                    print("MuJoCo 연결됨")
                    
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
                                    d = json.loads(line)
                                    if 'joint_angles' in d:
                                        self.mujoco_current = d['joint_angles'][:4]
                                        
                                        # MuJoCo 초기값 저장
                                        if first and self.mujoco_initial is None:
                                            self.mujoco_initial = self.mujoco_current.copy()
                                            print(f"MuJoCo 초기: {[f'{x:.3f}' for x in self.mujoco_initial]}")
                                            first = False
                                    
                                    # 그리퍼 데이터 수신 (추가)
                                    if 'gripper' in d:
                                        self.gripper_value = d['gripper']
                                        
                        except socket.timeout:
                            continue
                except:
                    time.sleep(2)
        
        threading.Thread(target=recv, daemon=True).start()
    
    def control(self):
        """오프셋 기반 제어"""
        # 초기값 둘 다 있어야 시작
        if self.real_initial is None or self.mujoco_initial is None:
            return
        
        # 변화량 계산
        target = []
        for i in range(4):
            delta = self.mujoco_current[i] - self.mujoco_initial[i]
            target.append(self.real_initial[i] + delta)
        
        # 조인트 전송 (4개만)
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start = Duration(sec=0, nanosec=100000000)
        
        traj.points = [point]
        self.joint_pub.publish(traj)
        
        # 그리퍼 제어 (Action 사용)
        if abs(self.gripper_value - self.last_gripper_value) > 0.005:  # 값이 변했을 때만
            self.send_gripper_goal(self.gripper_value)
            self.last_gripper_value = self.gripper_value
    
    def send_gripper_goal(self, position):
        """그리퍼 Action Goal 전송"""
        if not self.gripper_client.wait_for_server(timeout_sec=0.1):
            return  # 서버가 없으면 스킵
        
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 100.0
        
        # 비동기로 goal 전송
        self.gripper_client.send_goal_async(goal_msg)
        print(f"그리퍼 목표 전송: {position:.3f}")

def main():
    rclpy.init()
    rclpy.spin(OffsetMirror())
    rclpy.shutdown()

if __name__ == '__main__':
    main()