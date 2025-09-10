#!/usr/bin/env python3
"""
VR 왼쪽 컨트롤러 포즈 수집 (Docker에서 실행) - 수정 버전
Host IP를 직접 지정할 수 있도록 개선
"""

import rospy
import socket
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans
import sys

class VRPoseCollector:
    def __init__(self):
        rospy.init_node('vr_pose_collector')
        
        print("🎮 VR Left Controller Pose Collector (Fixed)")
        print("📡 왼쪽 컨트롤러 데이터 수집 중...\n")
        
        # VR 데이터
        self.vr_position = [0.0, 0.0, 0.0]
        self.vr_orientation = [0.0, 0.0, 0.0, 1.0]
        self.vr_trigger = 0.0
        self.calibrated = False
        
        # 초기 캘리브레이션 값
        self.initial_position = None
        self.initial_orientation = None
        
        # 데이터 수신 카운터
        self.data_count = 0
        
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
        # 여러 Host 주소 시도
        host_addresses = [
            'host.docker.internal',  # Docker Desktop
            '172.17.0.1',           # Docker 기본 브릿지
            '192.168.1.100',         # 실제 Host IP (필요시 수정)
            'localhost'              # 로컬 (같은 네트워크 namespace인 경우)
        ]
        
        # 명령줄 인자로 Host IP 지정 가능
        if len(sys.argv) > 1:
            host_addresses.insert(0, sys.argv[1])
            print(f"🎯 지정된 Host IP: {sys.argv[1]}")
        
        self.socket = None
        for host in host_addresses:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)  # 2초 타임아웃
                sock.connect((host, 12346))
                print(f"🔗 Host 연결 성공: {host}:12346")
                self.socket = sock
                break
            except Exception as e:
                print(f"⚠️ {host} 연결 실패: {e}")
        
        if not self.socket:
            print("❌ 모든 Host 주소 연결 실패")
            print("💡 Host IP를 직접 지정하세요:")
            print("   python3 vr_pose_collector_fixed.py <HOST_IP>")
    
    def setup_ros_topics(self):
        """ROS 토픽 설정"""
        # 왼쪽 컨트롤러 포즈
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, self.pose_callback)
        print("✅ /q2r_left_hand_pose 토픽 구독")
        
        # 왼쪽 컨트롤러 입력 (트리거 등)
        try:
            from quest2ros.msg import OVR2ROSInputs
            rospy.Subscriber('/q2r_left_hand_inputs', OVR2ROSInputs, self.input_callback)
            print("✅ /q2r_left_hand_inputs 토픽 구독")
        except ImportError:
            print("⚠️ quest2ros 메시지 타입 없음 - 포즈만 수집")
    
    def pose_callback(self, msg):
        """VR 포즈 콜백"""
        self.data_count += 1
        
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
        
        # 저장
        self.vr_position = relative_pos.tolist()
        self.vr_orientation = current_ori.tolist()
        
        # Host로 전송
        self.send_to_host()
        
        # 디버그: 10번마다 출력
        if self.data_count % 10 == 0:
            print(f"📊 데이터 수신: {self.data_count}개")
    
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
                    'vr_data': {  # sync_data_recorder가 기대하는 키
                        'position': self.vr_position,
                        'orientation': self.vr_orientation,
                        'trigger': self.vr_trigger,
                        'calibrated': self.calibrated
                    },
                    'timestamp': rospy.Time.now().to_sec()
                }
                json_data = json.dumps(data) + '\n'
                self.socket.sendall(json_data.encode())
            except Exception as e:
                print(f"❌ 전송 실패: {e}")
                # 재연결 시도
                self.socket = None
                self.setup_socket()
    
    def print_status(self, event):
        """상태 출력"""
        if self.calibrated:
            pos_str = ', '.join([f'{p:.3f}' for p in self.vr_position])
            socket_status = "Socket✓" if self.socket else "Socket✗"
            print(f"📍 VR 위치: [{pos_str}] | 트리거: {self.vr_trigger:.2f} | {socket_status} | 데이터: {self.data_count}")
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
        pass
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == '__main__':
    main()