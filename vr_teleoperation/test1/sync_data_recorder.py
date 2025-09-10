#!/usr/bin/env python3
"""
통합 데이터 수집기 - Robot Joint와 VR Pose를 동기화하여 저장
Host에서 실행하여 모든 데이터를 수집
"""

import socket
import json
import time
import threading
from datetime import datetime
from collections import deque

class SyncDataRecorder:
    def __init__(self):
        print("\n📊 Sync Data Recorder - 통합 데이터 수집")
        print("=" * 50)
        print("수집 데이터:")
        print("  - 실물 로봇 Joint 값 (left_arm_teaching.py)")
        print("  - VR 컨트롤러 Pose (vr_pose_collector.py)")
        print("=" * 50)
        
        # 현재 데이터
        self.current_robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.current_robot_gripper = -0.01
        self.current_vr_position = [0.0, 0.0, 0.0]
        self.current_vr_orientation = [0.0, 0.0, 0.0, 1.0]
        self.current_vr_trigger = 0.0
        
        # 수집된 데이터
        self.collected_samples = []
        self.sample_count = 0
        
        # 데이터 수신 플래그
        self.robot_data_received = False
        self.vr_data_received = False
        
        # 소켓 서버 (두 개 포트)
        self.setup_socket_servers()
        
        # 상태 출력
        self.status_thread = threading.Thread(target=self.status_loop, daemon=True)
        self.status_thread.start()
        
        print("\n✅ 데이터 수집기 준비 완료")
        print("\n조작법:")
        print("  Space - 현재 매핑 저장")
        print("  S - 파일로 내보내기")
        print("  R - 수집 상태 리셋")
        print("  Q - 종료\n")
    
    def setup_socket_servers(self):
        """두 개의 소켓 서버 설정"""
        # 로봇 데이터 서버 (포트 12345 - MuJoCo와 공유)
        def robot_server():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', 12345))
            server.listen(2)  # MuJoCo와 Recorder 둘 다 연결
            
            print("📡 로봇 데이터 서버 시작 (포트 12345)")
            
            while True:
                try:
                    client, addr = server.accept()
                    threading.Thread(target=self.handle_robot_client, args=(client,), daemon=True).start()
                except Exception as e:
                    print(f"로봇 서버 오류: {e}")
        
        # VR 데이터 서버 (포트 12346)
        def vr_server():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', 12346))
            server.listen(1)
            
            print("📡 VR 데이터 서버 시작 (포트 12346)")
            
            while True:
                try:
                    client, addr = server.accept()
                    print(f"🔗 VR 클라이언트 연결: {addr}")
                    threading.Thread(target=self.handle_vr_client, args=(client,), daemon=True).start()
                except Exception as e:
                    print(f"VR 서버 오류: {e}")
        
        threading.Thread(target=robot_server, daemon=True).start()
        threading.Thread(target=vr_server, daemon=True).start()
    
    def handle_robot_client(self, client):
        """로봇 데이터 처리"""
        buffer = ""
        while True:
            try:
                data = client.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        try:
                            msg = json.loads(line)
                            if 'left_arm' in msg:
                                if 'joint_angles' in msg['left_arm']:
                                    self.current_robot_joints = msg['left_arm']['joint_angles'][:4]
                                    self.robot_data_received = True
                                if 'gripper' in msg['left_arm']:
                                    self.current_robot_gripper = msg['left_arm']['gripper']
                        except json.JSONDecodeError:
                            pass
            except Exception:
                break
        client.close()
    
    def handle_vr_client(self, client):
        """VR 데이터 처리"""
        buffer = ""
        while True:
            try:
                data = client.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        try:
                            msg = json.loads(line)
                            if 'vr_data' in msg:
                                vr = msg['vr_data']
                                if 'position' in vr:
                                    self.current_vr_position = vr['position']
                                    self.vr_data_received = True
                                if 'orientation' in vr:
                                    self.current_vr_orientation = vr['orientation']
                                if 'trigger' in vr:
                                    self.current_vr_trigger = vr['trigger']
                        except json.JSONDecodeError:
                            pass
            except Exception:
                break
        client.close()
    
    def save_sample(self):
        """현재 상태를 샘플로 저장"""
        if not (self.robot_data_received and self.vr_data_received):
            print("⚠️ 아직 모든 데이터가 준비되지 않았습니다")
            return
        
        sample = {
            'sample_id': self.sample_count,
            'timestamp': datetime.now().isoformat(),
            'robot_joints': self.current_robot_joints.copy(),
            'robot_gripper': self.current_robot_gripper,
            'vr_position': self.current_vr_position.copy(),
            'vr_orientation': self.current_vr_orientation.copy(),
            'vr_trigger': self.current_vr_trigger
        }
        
        self.collected_samples.append(sample)
        self.sample_count += 1
        
        print(f"\n💾 샘플 #{self.sample_count} 저장됨")
        print(f"   Robot: [{', '.join([f'{j:.2f}' for j in self.current_robot_joints])}]")
        print(f"   VR: [{', '.join([f'{p:.3f}' for p in self.current_vr_position])}]")
    
    def save_to_file(self):
        """수집한 데이터를 파일로 저장"""
        if not self.collected_samples:
            print("⚠️ 저장할 데이터가 없습니다")
            return
        
        filename = f"vr_joint_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'total_samples': len(self.collected_samples),
                'robot_type': 'OpenManipulator-X',
                'arm': 'left',
                'collection_method': 'reverse_teaching'
            },
            'samples': self.collected_samples
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ {len(self.collected_samples)}개 샘플을 {filename}에 저장")
        return filename
    
    def status_loop(self):
        """상태 출력 루프"""
        while True:
            time.sleep(2)
            
            print("\n" + "=" * 50)
            print(f"📊 수집 상태: {self.sample_count}개 샘플")
            print(f"   로봇 데이터: {'✅' if self.robot_data_received else '⏳ 대기'}")
            print(f"   VR 데이터: {'✅' if self.vr_data_received else '⏳ 대기'}")
            
            if self.robot_data_received:
                joints_str = ', '.join([f'{j:.2f}' for j in self.current_robot_joints])
                print(f"   현재 Joint: [{joints_str}]")
            
            if self.vr_data_received:
                pos_str = ', '.join([f'{p:.3f}' for p in self.current_vr_position])
                print(f"   현재 VR: [{pos_str}]")
    
    def run(self):
        """메인 루프"""
        try:
            while True:
                cmd = input("\n명령 (space/s/r/q): ").strip().lower()
                
                if cmd == ' ' or cmd == 'space':
                    self.save_sample()
                elif cmd == 's':
                    self.save_to_file()
                elif cmd == 'r':
                    self.collected_samples = []
                    self.sample_count = 0
                    print("🔄 수집 데이터 리셋")
                elif cmd == 'q':
                    # 자동 저장
                    if self.collected_samples:
                        self.save_to_file()
                    break
        
        except KeyboardInterrupt:
            print("\n\n중단됨")
            if self.collected_samples:
                self.save_to_file()

def main():
    recorder = SyncDataRecorder()
    recorder.run()
    print("\n🏁 데이터 수집 종료")

if __name__ == '__main__':
    main()