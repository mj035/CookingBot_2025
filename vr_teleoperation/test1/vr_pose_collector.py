#!/usr/bin/env python3
"""
VR 왼쪽 컨트롤러 포즈 수집 (Docker에서 실행)
ROS1 토픽에서 VR 데이터를 수집하여 Host로 전송
"""

import rospy
import socket
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans

class VRPoseCollector:
    def __init__(self):
        rospy.init_node('vr_pose_collector')
        
        print("🎮 VR Left Controller Pose Collector")
        print("📡 왼쪽 컨트롤러 데이터 수집 중...\n")
        
        # VR 데이터
        self.vr_position = [0.0, 0.0, 0.0]
        self.vr_orientation = [0.0, 0.0, 0.0, 1.0]
        self.vr_trigger = 0.0
        self.calibrated = False
        
        # 초기 캘리브레이션 값
        self.initial_position = None
        self.initial_orientation = None
        
        # Host로 전송할 소켓
        self.setup_socket()
        
        # ROS 토픽 구독
        self.setup_ros_topics()
        
        # 상태 출력 타이머
        rospy.Timer(rospy.Duration(1.0), self.print_status)
        
        print("✅ VR 수집기 준비 완료")
        print("🎯 VR 컨트롤러를 편안한 위치에 두고 시작하세요\n")
    
    def setup_socket(self):
        """Host로 데이터 전송할 소켓"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Host의 sync_data_recorder.py에 연결 (포트 12346)
            self.socket.connect(('host.docker.internal', 12346))  # Docker → Host
            print("🔗 Host 데이터 수집기 연결 성공")
        except:
            print("⚠️ Host 연결 실패 - 로컬 로깅만 진행")
            self.socket = None
    
    def setup_ros_topics(self):
        """ROS 토픽 설정"""
        # 왼쪽 컨트롤러 포즈
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, self.pose_callback)
        
        # 왼쪽 컨트롤러 입력 (트리거 등)
        try:
            from quest2ros.msg import OVR2ROSInputs
            rospy.Subscriber('/q2r_left_hand_inputs', OVR2ROSInputs, self.input_callback)
            print("✅ VR 입력 토픽 구독")
        except ImportError:
            print("⚠️ quest2ros 메시지 타입 없음 - 포즈만 수집")
    
    def pose_callback(self, msg):
        """VR 포즈 콜백"""
        # 현재 위치와 방향
        current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        current_ori = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        # 초기 캘리브레이션
        if not self.calibrated:
            self.initial_position = current_pos.copy()
            self.initial_orientation = current_ori.copy()
            self.calibrated = True
            print("🎯 VR 캘리브레이션 완료!")
        
        # 상대 위치 계산 (오프셋)
        relative_pos = current_pos - self.initial_position
        
        # 상대 방향 계산
        current_euler = tf_trans.euler_from_quaternion(current_ori)
        initial_euler = tf_trans.euler_from_quaternion(self.initial_orientation)
        relative_euler = np.array(current_euler) - np.array(initial_euler)
        
        # 저장
        self.vr_position = relative_pos.tolist()
        self.vr_orientation = current_ori.tolist()  # Quaternion 그대로 저장
        
        # Host로 전송
        self.send_to_host()
    
    def input_callback(self, msg):
        """VR 입력 콜백"""
        if hasattr(msg, 'trigger'):
            self.vr_trigger = msg.trigger
        
        # A+B 버튼으로 재캘리브레이션
        if hasattr(msg, 'button_upper') and hasattr(msg, 'button_lower'):
            if msg.button_upper and msg.button_lower:
                self.recalibrate()
    
    def recalibrate(self):
        """재캘리브레이션"""
        self.calibrated = False
        print("🔄 재캘리브레이션 요청됨")
    
    def send_to_host(self):
        """Host로 VR 데이터 전송"""
        if self.socket and self.calibrated:
            try:
                data = {
                    'vr_data': {
                        'position': self.vr_position,
                        'orientation': self.vr_orientation,
                        'trigger': self.vr_trigger,
                        'calibrated': self.calibrated
                    },
                    'timestamp': rospy.Time.now().to_sec()
                }
                json_data = json.dumps(data) + '\n'
                self.socket.sendall(json_data.encode())
            except:
                # 연결 끊김 무시
                pass
    
    def print_status(self, event):
        """상태 출력"""
        if self.calibrated:
            pos_str = ', '.join([f'{p:.3f}' for p in self.vr_position])
            print(f"📍 VR 위치: [{pos_str}] | 트리거: {self.vr_trigger:.2f}")
        else:
            print("⏳ VR 캘리브레이션 대기 중...")
    
    def run(self):
        """메인 루프"""
        rospy.spin()

def main():
    try:
        collector = VRPoseCollector()
        collector.run()
    except rospy.ROSInterruptException:
        print("\n종료")
    except KeyboardInterrupt:
        print("\n중단됨")

if __name__ == '__main__':
    main()