#!/usr/bin/env python3
"""
동기화된 데이터 레코더 - MuJoCo 조인트값 + VR 포즈값 수집
포트: 12347(MuJoCo), 12346(VR)
"""

import socket
import json
import time
import threading
from datetime import datetime

class SyncDataRecorder:
    def __init__(self):
        print("\n📊 Synchronized Data Recorder")
        print("=" * 50)
        print("포트 설정:")
        print("  - 12347: MuJoCo 조인트값 수신")
        print("  - 12346: VR 포즈값 수신")
        print("=" * 50)
        
        # 데이터 상태
        self.mujoco_joints = [0.0, 0.0, 0.0, 0.0]
        self.mujoco_gripper = 0.0
        self.vr_position = [0.0, 0.0, 0.0]
        self.vr_orientation = [0.0, 0.0, 0.0, 1.0]
        self.vr_trigger = 0.0
        
        # 연결 상태
        self.mujoco_connected = False
        self.vr_connected = False
        
        # 수집된 데이터
        self.collected_samples = []
        
        # 소켓 서버 설정
        self.setup_servers()
        
        print("\n📍 조작법:")
        print("  Space - 현재 상태 저장")
        print("  S - 파일로 내보내기")
        print("  C - 저장된 샘플 지우기")
        print("  Q - 종료\n")
    
    def setup_servers(self):
        """소켓 서버 설정"""
        # MuJoCo 데이터 서버 (포트 12347)
        def mujoco_server():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                server.bind(('localhost', 12347))
                server.listen(2)
                print("✅ MuJoCo 데이터 서버 시작 (포트 12347)")
            except Exception as e:
                print(f"❌ MuJoCo 서버 시작 실패: {e}")
                return
            
            while True:
                try:
                    client, addr = server.accept()
                    print(f"🎮 MuJoCo 연결: {addr}")
                    self.mujoco_connected = True
                    threading.Thread(target=self.handle_mujoco_data, 
                                   args=(client,), daemon=True).start()
                except Exception as e:
                    print(f"MuJoCo 서버 오류: {e}")
        
        # VR 데이터 서버 (포트 12346)
        def vr_server():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                server.bind(('0.0.0.0', 12346))  # Docker 접근 허용
                server.listen(1)
                print("✅ VR 데이터 서버 시작 (포트 12346)")
            except Exception as e:
                print(f"❌ VR 서버 시작 실패: {e}")
                return
            
            while True:
                try:
                    client, addr = server.accept()
                    print(f"🥽 VR 연결: {addr}")
                    self.vr_connected = True
                    threading.Thread(target=self.handle_vr_data, 
                                   args=(client,), daemon=True).start()
                except Exception as e:
                    print(f"VR 서버 오류: {e}")
        
        # 서버 스레드 시작
        threading.Thread(target=mujoco_server, daemon=True).start()
        threading.Thread(target=vr_server, daemon=True).start()
    
    def handle_mujoco_data(self, client):
        """MuJoCo 데이터 처리"""
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
                            if 'mujoco' in msg:
                                if 'joint_angles' in msg['mujoco']:
                                    self.mujoco_joints = msg['mujoco']['joint_angles'][:4]
                                if 'gripper' in msg['mujoco']:
                                    self.mujoco_gripper = msg['mujoco']['gripper']
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                if self.mujoco_connected:
                    print(f"⚠️ MuJoCo 데이터 오류: {e}")
                break
        
        self.mujoco_connected = False
        client.close()
        print("⚠️ MuJoCo 연결 끊김")
    
    def handle_vr_data(self, client):
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
                            if 'vr_controller' in msg:
                                controller = msg['vr_controller']
                                if 'position' in controller:
                                    self.vr_position = controller['position']
                                if 'orientation' in controller:
                                    self.vr_orientation = controller['orientation']
                                if 'trigger' in controller:
                                    self.vr_trigger = controller['trigger']
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                if self.vr_connected:
                    print(f"⚠️ VR 데이터 오류: {e}")
                break
        
        self.vr_connected = False
        client.close()
        print("⚠️ VR 연결 끊김")
    
    def save_current_state(self):
        """현재 상태 저장"""
        sample = {
            'sample_id': len(self.collected_samples) + 1,
            'timestamp': datetime.now().isoformat(),
            'mujoco': {
                'joints': self.mujoco_joints.copy(),
                'gripper': self.mujoco_gripper
            },
            'vr': {
                'position': self.vr_position.copy(),
                'orientation': self.vr_orientation.copy(),
                'trigger': self.vr_trigger
            },
            'sync_status': {
                'mujoco_connected': self.mujoco_connected,
                'vr_connected': self.vr_connected
            }
        }
        
        self.collected_samples.append(sample)
        
        # 상태 표시
        status = []
        if self.mujoco_connected:
            status.append("MuJoCo✓")
        else:
            status.append("MuJoCo✗")
        if self.vr_connected:
            status.append("VR✓")
        else:
            status.append("VR✗")
        
        print(f"\n💾 샘플 #{len(self.collected_samples)} 저장됨 [{' '.join(status)}]")
        
        # 데이터 미리보기
        joints_str = ', '.join([f'{j:.2f}' for j in self.mujoco_joints])
        vr_pos_str = ', '.join([f'{p:.3f}' for p in self.vr_position])
        print(f"  MuJoCo: [{joints_str}]")
        print(f"  VR: [{vr_pos_str}]")
    
    def save_to_file(self):
        """데이터를 파일로 저장"""
        if not self.collected_samples:
            print("⚠️ 저장할 데이터가 없습니다")
            return
        
        filename = f"mujoco_vr_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 요약 정보 추가
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.collected_samples),
                'mujoco_samples': sum(1 for s in self.collected_samples if s['sync_status']['mujoco_connected']),
                'vr_samples': sum(1 for s in self.collected_samples if s['sync_status']['vr_connected']),
                'synced_samples': sum(1 for s in self.collected_samples 
                                    if s['sync_status']['mujoco_connected'] and s['sync_status']['vr_connected'])
            },
            'samples': self.collected_samples
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ {len(self.collected_samples)}개 샘플을 {filename}에 저장")
        print(f"  - MuJoCo 데이터: {data['metadata']['mujoco_samples']}개")
        print(f"  - VR 데이터: {data['metadata']['vr_samples']}개")
        print(f"  - 동기화됨: {data['metadata']['synced_samples']}개")
    
    def print_status(self):
        """현재 상태 출력"""
        joints_str = ', '.join([f'{j:.2f}' for j in self.mujoco_joints])
        vr_pos_str = ', '.join([f'{p:.3f}' for p in self.vr_position])
        
        status = []
        if self.mujoco_connected:
            status.append("M✓")
        else:
            status.append("M✗")
        if self.vr_connected:
            status.append("V✓")
        else:
            status.append("V✗")
        
        print(f"\r[{' '.join(status)}] MuJoCo:[{joints_str}] VR:[{vr_pos_str}] 샘플:{len(self.collected_samples)}", end='')
    
    def run(self):
        """메인 루프"""
        print("시스템 준비 완료. 데이터 수집을 시작하세요.\n")
        
        # 상태 출력 스레드
        def status_thread():
            while True:
                self.print_status()
                time.sleep(0.1)
        
        threading.Thread(target=status_thread, daemon=True).start()
        
        # 키 입력 처리
        while True:
            try:
                cmd = input("\n\n명령 (space/s/c/q): ").strip().lower()
                
                if cmd == ' ' or cmd == 'space':
                    self.save_current_state()
                elif cmd == 's':
                    self.save_to_file()
                elif cmd == 'c':
                    self.collected_samples = []
                    print("🗑 샘플 데이터 초기화됨")
                elif cmd == 'q':
                    break
                    
            except KeyboardInterrupt:
                break
        
        # 자동 저장
        if self.collected_samples:
            print("\n자동 저장 중...")
            self.save_to_file()
        
        print("\n👋 프로그램 종료")

def main():
    recorder = SyncDataRecorder()
    recorder.run()

if __name__ == '__main__':
    main()