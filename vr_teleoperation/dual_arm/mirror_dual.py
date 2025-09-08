#!/usr/bin/env python3
"""
🤖 Dual-Arm VR → Physical Robot Bridge (Host/ROS2)

이 파일은 VR 컨트롤러의 움직임을 두 개의 실제 OpenManipulator-X 로봇에 전달하는 
핵심 브릿지 역할을 합니다.

주요 기능:
- Docker(ROS1)에서 Socket으로 받은 MuJoCo 양팔 조인트 값을 처리
- Offset-based Control: 절대 위치가 아닌 상대적 변화량만 적용
- 양팔 안전한 로봇 제어: 위험한 포즈 방지 및 부드러운 움직임 보장
- ROS2 JointTrajectory 메시지로 두 로봇에 명령 전송
- Gripper Action Client로 양쪽 그리퍼 제어

동작 방식:
왼쪽: target_joint = left_initial + (mujoco_left_current - mujoco_left_initial)
오른쪽: target_joint = right_initial + (mujoco_right_current - mujoco_right_initial)

Safety Features:
- Joint limit 체크 (양팔)
- 과도한 움직임 제한
- Emergency stop 기능
- 개별 팔 제어 가능
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

class DualArmOffsetMirror(Node):
    def __init__(self):
        super().__init__('dual_arm_offset_mirror')
        
        print("\n🤖 양팔 오프셋 미러링 - 변화량만 적용")
        
        # 양팔 초기값 저장
        self.robot_initial = {
            'left': None,   # 왼쪽 실물 초기 위치 
            'right': None   # 오른쪽 실물 초기 위치
        }
        
        self.mujoco_initial = {
            'left': None,   # MuJoCo 왼쪽 초기 위치
            'right': None   # MuJoCo 오른쪽 초기 위치
        }
        
        self.mujoco_current = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }
        
        self.gripper_values = {
            'left': -0.01,
            'right': -0.01
        }
        
        # 로봇 상태 추적
        self.robot_status = {
            'left_connected': False,
            'right_connected': False,
            'left_control_count': 0,
            'right_control_count': 0,
            'last_left_time': 0.0,
            'last_right_time': 0.0
        }
        
        # 그리퍼 마지막 값 추적 (중복 전송 방지)
        self.last_gripper_values = {
            'left': -0.01,
            'right': -0.01
        }
        
        # ROS2 Publishers (양팔)
        # 왼쪽 로봇 (기존 설정, 모터 ID 11-15)
        self.left_joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        
        # 오른쪽 로봇 (새로운 설정, 다른 모터 ID)
        self.right_joint_pub = self.create_publisher(
            JointTrajectory, '/right_arm_controller/joint_trajectory', 10)
        
        # Joint State Subscriber (단일 토픽에서 양팔 데이터)
        # 하나의 토픽에서 모든 조인트 (left: joint1~4, right: right_joint1~4)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        
        # Gripper Action Clients (양팔)
        self.left_gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(
            self, GripperCommand, '/right_gripper_controller/gripper_cmd')
        
        # MuJoCo 연결
        self.setup_socket()
        
        # 제어 루프 (양팔 통합)
        self.timer = self.create_timer(0.05, self.dual_arm_control)  # 20Hz
        
        # 상태 모니터링 타이머
        self.status_timer = self.create_timer(3.0, self.print_status)
        
        print("🤖 양팔 초기 위치 읽는 중...")
        print("📡 통합 토픽: /joint_states (모든 조인트 포함)")
    
    def joint_states_callback(self, msg):
        """통합 Joint States 콜백 (양팔 데이터 처리)"""
        # 왼쪽 로봇 초기값 저장
        if self.robot_initial['left'] is None:
            self.robot_initial['left'] = []
            for name in ['joint1', 'joint2', 'joint3', 'joint4']:
                for i, n in enumerate(msg.name):
                    if name == n:  # 정확히 일치하는 이름
                        self.robot_initial['left'].append(msg.position[i])
                        break
            
            if len(self.robot_initial['left']) == 4:
                print(f"✅ 왼쪽 로봇 초기값: {[f'{x:.3f}' for x in self.robot_initial['left']]}")
                self.robot_status['left_connected'] = True
        
        # 오른쪽 로봇 초기값 저장
        if self.robot_initial['right'] is None:
            self.robot_initial['right'] = []
            for name in ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']:
                for i, n in enumerate(msg.name):
                    if name == n:  # 정확히 일치하는 이름
                        self.robot_initial['right'].append(msg.position[i])
                        break
            
            if len(self.robot_initial['right']) == 4:
                print(f"✅ 오른쪽 로봇 초기값: {[f'{x:.3f}' for x in self.robot_initial['right']]}")
                self.robot_status['right_connected'] = True
    
    def setup_socket(self):
        """MuJoCo 소켓 연결 설정 (양팔 데이터 수신)"""
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
                                        
                                        # 왼쪽 팔 데이터 처리
                                        if 'left_arm' in d and 'joint_angles' in d['left_arm']:
                                            self.mujoco_current['left'] = d['left_arm']['joint_angles'][:4]
                                            
                                            # 왼쪽 MuJoCo 초기값 저장
                                            if first and self.mujoco_initial['left'] is None:
                                                self.mujoco_initial['left'] = self.mujoco_current['left'].copy()
                                                print(f"✅ MuJoCo 왼쪽 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['left']]}")
                                        
                                        # 왼쪽 그리퍼 데이터
                                        if 'left_arm' in d and 'gripper' in d['left_arm']:
                                            self.gripper_values['left'] = d['left_arm']['gripper']
                                        
                                        # 오른쪽 팔 데이터 처리
                                        if 'right_arm' in d and 'joint_angles' in d['right_arm']:
                                            self.mujoco_current['right'] = d['right_arm']['joint_angles'][:4]
                                            
                                            # 오른쪽 MuJoCo 초기값 저장
                                            if first and self.mujoco_initial['right'] is None:
                                                self.mujoco_initial['right'] = self.mujoco_current['right'].copy()
                                                print(f"✅ MuJoCo 오른쪽 초기값: {[f'{x:.3f}' for x in self.mujoco_initial['right']]}")
                                                first = False
                                        
                                        # 오른쪽 그리퍼 데이터
                                        if 'right_arm' in d and 'gripper' in d['right_arm']:
                                            self.gripper_values['right'] = d['right_arm']['gripper']
                                        
                                    except json.JSONDecodeError:
                                        continue
                        except socket.timeout:
                            continue
                except Exception as e:
                    print(f"⚠️ MuJoCo 연결 오류: {e}")
                    time.sleep(2)
        
        # 소켓 수신 스레드 시작
        threading.Thread(target=recv, daemon=True).start()
    
    def create_joint_trajectory(self, target_joints, arm_side='left'):
        """조인트 궤적 메시지 생성"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        
        # 팔별 조인트 이름 설정
        if arm_side == 'right':
            traj.joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4']
        else:
            traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms
        
        traj.points = [point]
        return traj
    
    def apply_safety_limits(self, joints, arm_side='left'):
        """안전 제한 적용"""
        # 조인트별 안전 범위
        joint_limits = [
            [-3.14, 3.14],   # Joint 1
            [-1.5, 1.5],     # Joint 2  
            [-1.5, 1.4],     # Joint 3
            [-1.7, 1.97]     # Joint 4
        ]
        
        safe_joints = []
        for i, (joint_val, limits) in enumerate(zip(joints, joint_limits)):
            # 급격한 변화 제한 (0.1 라디안/스텝)
            max_change = 0.1  # 라디안
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
    
    def send_gripper_goal(self, position, arm_side='left'):
        """그리퍼 Action Goal 전송"""
        client = self.left_gripper_client if arm_side == 'left' else self.right_gripper_client
        
        if not client.wait_for_server(timeout_sec=0.5):  # 타임아웃 증가
            print(f"⚠️ {arm_side.upper()} 그리퍼 서버 연결 실패")
            return
        
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)  # float 명시적 변환
        goal_msg.command.max_effort = 100.0
        
        # 비동기로 goal 전송
        future = client.send_goal_async(goal_msg)
        # print(f"{arm_side.upper()} 그리퍼: {position:.3f}")  # 디버그 출력 간소화
    
    def dual_arm_control(self):
        """양팔 오프셋 기반 제어"""
        current_time = time.time()
        
        # 왼쪽 팔 제어
        if (self.robot_initial['left'] is not None and 
            self.mujoco_initial['left'] is not None):
            
            # 오프셋 계산
            left_target = []
            for i in range(4):
                delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                target_val = self.robot_initial['left'][i] + delta
                left_target.append(target_val)
            
            # 안전 제한 적용
            safe_left_target = self.apply_safety_limits(left_target, 'left')
            
            # 궤적 생성 및 전송
            left_traj = self.create_joint_trajectory(safe_left_target, 'left')
            self.left_joint_pub.publish(left_traj)
            
            # 왼쪽 그리퍼 제어
            if abs(self.gripper_values['left'] - self.last_gripper_values['left']) > 0.002:  # 더 민감하게
                self.send_gripper_goal(self.gripper_values['left'], 'left')
                self.last_gripper_values['left'] = self.gripper_values['left']
            
            self.robot_status['left_control_count'] += 1
            self.robot_status['last_left_time'] = current_time
        
        # 오른쪽 팔 제어
        if (self.robot_initial['right'] is not None and 
            self.mujoco_initial['right'] is not None):
            
            # 오프셋 계산
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
            
            # 오른쪽 그리퍼 제어
            if abs(self.gripper_values['right'] - self.last_gripper_values['right']) > 0.002:  # 더 민감하게
                self.send_gripper_goal(self.gripper_values['right'], 'right')
                self.last_gripper_values['right'] = self.gripper_values['right']
            
            self.robot_status['right_control_count'] += 1
            self.robot_status['last_right_time'] = current_time
    
    def print_status(self):
        """상태 정보 출력"""
        print(f"\n🤖 === 양팔 미러링 상태 ===")
        print(f"🔗 연결: 왼쪽={'✅' if self.robot_status['left_connected'] else '❌'} 오른쪽={'✅' if self.robot_status['right_connected'] else '❌'}")
        
        # 초기화 상태
        left_ready = (self.robot_initial['left'] is not None and 
                     self.mujoco_initial['left'] is not None)
        right_ready = (self.robot_initial['right'] is not None and 
                      self.mujoco_initial['right'] is not None)
        
        print(f"🎯 제어 준비: 왼쪽={'✅' if left_ready else '❌'} 오른쪽={'✅' if right_ready else '❌'}")
        print(f"📊 제어 횟수: 왼쪽={self.robot_status['left_control_count']} 오른쪽={self.robot_status['right_control_count']}")
        
        # 그리퍼 상태
        print(f"🖐 그리퍼 값: 왼쪽={self.gripper_values['left']:.3f} 오른쪽={self.gripper_values['right']:.3f}")
        
        # 현재 목표 조인트 출력
        if left_ready:
            left_target = []
            for i in range(4):
                delta = self.mujoco_current['left'][i] - self.mujoco_initial['left'][i]
                target = self.robot_initial['left'][i] + delta
                left_target.append(target)
            print(f"🎯 왼쪽 목표: {[f'{x:.3f}' for x in left_target]}")
        
        if right_ready:
            right_target = []  
            for i in range(4):
                delta = self.mujoco_current['right'][i] - self.mujoco_initial['right'][i]
                target = self.robot_initial['right'][i] + delta
                right_target.append(target)
            print(f"🎯 오른쪽 목표: {[f'{x:.3f}' for x in right_target]}")
        
        current_time = time.time()
        print(f"⏰ 최근 제어: 왼쪽={current_time - self.robot_status['last_left_time']:.1f}초 전, "
              f"오른쪽={current_time - self.robot_status['last_right_time']:.1f}초 전")
    
    def emergency_stop(self):
        """비상 정지"""
        print("🛑 비상 정지 실행!")
        
        # 현재 위치 유지 명령 전송
        if self.robot_initial['left'] is not None:
            stop_traj = self.create_joint_trajectory(self.robot_initial['left'], 'left')
            self.left_joint_pub.publish(stop_traj)
        
        if self.robot_initial['right'] is not None:
            stop_traj = self.create_joint_trajectory(self.robot_initial['right'], 'right')
            self.right_joint_pub.publish(stop_traj)
        
        # 그리퍼도 현재 위치 유지
        self.send_gripper_goal(self.last_gripper_values['left'], 'left')
        self.send_gripper_goal(self.last_gripper_values['right'], 'right')
    
    def reset_calibration(self, arm_side='both'):
        """캘리브레이션 리셋"""
        if arm_side in ['left', 'both']:
            self.robot_initial['left'] = None
            self.mujoco_initial['left'] = None
            self.robot_status['left_connected'] = False
            print("🔄 왼쪽 캘리브레이션 리셋")
        
        if arm_side in ['right', 'both']:
            self.robot_initial['right'] = None
            self.mujoco_initial['right'] = None
            self.robot_status['right_connected'] = False
            print("🔄 오른쪽 캘리브레이션 리셋")

def main():
    rclpy.init()
    
    try:
        dual_mirror = DualArmOffsetMirror()
        
        print("\n🤖 === 양팔 오프셋 미러링 시스템 ===")
        print("🎯 왼쪽 VR 컨트롤러 → 왼쪽 OpenManipulator-X (모터 ID 11~15)")
        print("🎯 오른쪽 VR 컨트롤러 → 오른쪽 OpenManipulator-X (모터 ID 21~25)") 
        print("📡 필요한 ROS2 토픽:")
        print("   - /arm_controller/joint_trajectory (왼팔)")
        print("   - /right_arm_controller/joint_trajectory (오른팔)")  
        print("   - /gripper_controller/gripper_cmd (왼쪽 그리퍼)")
        print("   - /right_gripper_controller/gripper_cmd (오른쪽 그리퍼)")
        print("   - /joint_states (양팔 통합)")
        print("⚠️  양쪽 로봇이 모두 연결되어야 제어 시작됩니다")
        print("🖐 그리퍼는 VR 트리거로 제어됩니다")
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
        print("🏁 양팔 미러링 시스템 종료")

if __name__ == '__main__':
    main()