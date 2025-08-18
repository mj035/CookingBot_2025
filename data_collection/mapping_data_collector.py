#!/usr/bin/env python3
"""
🎯 MuJoCo Mapping Data Collector - 터미널 키보드 입력 + 데이터 수집
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
import sys
import select
import termios
import tty
import socket
import json
import pickle
import os

class MappingDataCollector:
    def __init__(self):
        print("🎯 매핑 데이터 수집기 초기화 중...")
        
        # 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # 제어 변수
        self.selected_joint = 0
        self.joint_step = 0.05
        self.running = True
        
        # 데이터 수집
        self.data_collection_mode = False
        self.collected_mappings = []
        self.collection_count = 0
        
        # VR 브릿지 연결
        self.socket = None
        self.connected = False
        self.vr_position_delta = None
        self.connect_to_vr_bridge()
        
        # 액추에이터 매핑
        self.joint_mapping = {}
        self.setup_actuator_mapping()
        self.reset_robot_pose()
        
        # 키보드 입력
        self.key_pressed = None
        self.old_settings = None
        
    def connect_to_vr_bridge(self):
        """VR 브릿지 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', 12345))
            self.socket.settimeout(0.001)
            self.connected = True
            print("✅ VR 브릿지 연결 성공!")
            
            # VR 데이터 수신 스레드
            threading.Thread(target=self.vr_data_thread, daemon=True).start()
            
        except Exception as e:
            print(f"⚠️ VR 브릿지 연결 실패: {e} - 수동 모드로 시작")
            self.connected = False
    
    def vr_data_thread(self):
        """VR 데이터 수신 스레드"""
        data_buffer = ""
        while self.connected and self.running:
            try:
                raw_data = self.socket.recv(4096).decode('utf-8', errors='ignore')
                if not raw_data:
                    continue
                
                data_buffer += raw_data
                while '\n' in data_buffer:
                    line, data_buffer = data_buffer.split('\n', 1)
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # VR 위치 델타 추출
                            if 'vr_position_delta' in data:
                                self.vr_position_delta = data['vr_position_delta']
                        except:
                            continue
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ VR 데이터 오류: {e}")
                self.connected = False
                break
            
            time.sleep(0.001)
    
    def get_vr_position_delta(self):
        """현재 VR 위치 델타 반환"""
        if self.connected and hasattr(self, 'vr_position_delta') and self.vr_position_delta:
            return self.vr_position_delta
        else:
            # VR 연결 안된 경우 더미 데이터
            return [0.1, 0.05, 0.15]  # 예시 위치
    
    def setup_actuator_mapping(self):
        """액추에이터 매핑"""
        for joint_idx in range(4):
            pattern = f'actuator_joint{joint_idx+1}'
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern)
                self.joint_mapping[joint_idx] = actuator_id
                print(f"✅ Joint{joint_idx+1} → 액추에이터 {actuator_id}")
            except:
                self.joint_mapping[joint_idx] = -1
    
    def reset_robot_pose(self):
        """로봇 초기 자세"""
        initial_pose = [0.0, -0.2, 0.3, -0.8]
        for joint_idx, angle in enumerate(initial_pose):
            actuator_id = self.joint_mapping.get(joint_idx, -1)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = angle
        
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)
        print("✅ 로봇 초기 자세 설정 완료")
    
    def get_current_joint_values(self):
        """현재 조인트 값 반환"""
        joints = []
        for i in range(4):
            actuator_id = self.joint_mapping.get(i, -1)
            if actuator_id >= 0:
                joints.append(self.data.ctrl[actuator_id])
            else:
                joints.append(0.0)
        return joints
    
    def set_joint_value(self, joint_idx, value):
        """특정 조인트 값 설정"""
        joint_limits = [
            [-3.14, 3.14], [-2.0, 1.8], [-1.5, 1.4], [-2.0, 2.0]
        ]
        
        actuator_id = self.joint_mapping.get(joint_idx, -1)
        if actuator_id >= 0:
            safe_angle = np.clip(value, joint_limits[joint_idx][0], joint_limits[joint_idx][1])
            self.data.ctrl[actuator_id] = safe_angle
    
    def collect_mapping_data(self):
        """현재 VR 위치와 로봇 조인트로 매핑 데이터 수집"""
        if not self.data_collection_mode:
            print("❌ 데이터 수집 모드를 먼저 활성화하세요 (키: 'm')")
            return
        
        # VR 위치 델타 가져오기
        vr_pos_delta = self.get_vr_position_delta()
        
        # 현재 로봇 조인트 값
        current_joints = self.get_current_joint_values()
        
        # 매핑 데이터 생성
        mapping_data = {
            'vr_delta': vr_pos_delta,
            'joints': current_joints.copy(),
            'name': f'수집_{self.collection_count + 1}',
            'timestamp': time.time()
        }
        
        # 수집된 데이터에 추가
        self.collected_mappings.append(mapping_data)
        self.collection_count += 1
        
        print(f"✅ 매핑 데이터 #{self.collection_count} 수집 완료!")
        print(f"   VR 위치: [{vr_pos_delta[0]:+.3f}, {vr_pos_delta[1]:+.3f}, {vr_pos_delta[2]:+.3f}]")
        print(f"   조인트: [{', '.join([f'{j:.3f}' for j in current_joints])}]")
        
        # VR 브릿지에 알림 (연결된 경우)
        if self.connected:
            try:
                message = json.dumps({
                    'command': 'mapping_collected',
                    'count': self.collection_count,
                    'total_mappings': len(self.collected_mappings)
                }) + '\n'
                self.socket.sendall(message.encode())
            except:
                pass
    
    def save_collected_data(self):
        """수집된 데이터를 파일로 저장"""
        if not self.collected_mappings:
            print("❌ 저장할 데이터가 없습니다")
            return
        
        # 파일명 생성
        timestamp = int(time.time())
        filename = f'mapping_data_{timestamp}.pkl'
        backup_filename = f'mapping_data_{timestamp}_backup.json'
        
        try:
            # pickle 형태로 저장 (VR 브릿지와 호환)
            with open(filename, 'wb') as f:
                data = {'position': self.collected_mappings}
                pickle.dump(data, f)
            
            # JSON 백업도 저장
            with open(backup_filename, 'w') as f:
                json.dump(self.collected_mappings, f, indent=2)
            
            print(f"✅ 데이터 저장 완료!")
            print(f"   메인 파일: {filename}")
            print(f"   백업 파일: {backup_filename}")
            print(f"   총 {len(self.collected_mappings)}개 매핑 포인트")
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    def load_existing_data(self):
        """기존 데이터 로드"""
        try:
            # 가장 최근 파일 찾기
            mapping_files = [f for f in os.listdir('.') if f.startswith('mapping_data_') and f.endswith('.pkl')]
            if mapping_files:
                latest_file = sorted(mapping_files)[-1]
                with open(latest_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'position' in data:
                        self.collected_mappings = data['position']
                        self.collection_count = len(self.collected_mappings)
                        print(f"✅ 기존 데이터 로드: {self.collection_count}개 매핑")
        except Exception as e:
            print(f"⚠️ 기존 데이터 로드 실패: {e}")
    
    def keyboard_input_thread(self):
        """키보드 입력 스레드"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
                    self.key_pressed = key
                time.sleep(0.01)
                
        except Exception as e:
            print(f"키보드 입력 오류: {e}")
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def handle_keyboard_input(self):
        """키보드 입력 처리"""
        if self.key_pressed is None:
            return
        
        key = self.key_pressed
        self.key_pressed = None
        
        current_joints = self.get_current_joint_values()
        
        # 조인트 선택
        if key in ['1', '2', '3', '4']:
            self.selected_joint = int(key) - 1
            print(f"🎮 Joint{self.selected_joint+1} 선택됨 (현재값: {current_joints[self.selected_joint]:.3f})")
        
        # 조인트 조작
        elif key in ['+', '=']:
            new_value = current_joints[self.selected_joint] + self.joint_step
            self.set_joint_value(self.selected_joint, new_value)
            print(f"🔺 Joint{self.selected_joint+1}: {new_value:.3f}")
        
        elif key == '-':
            new_value = current_joints[self.selected_joint] - self.joint_step
            self.set_joint_value(self.selected_joint, new_value)
            print(f"🔻 Joint{self.selected_joint+1}: {new_value:.3f}")
        
        # 정밀 조작
        elif key == 'q':
            new_value = current_joints[self.selected_joint] + 0.01
            self.set_joint_value(self.selected_joint, new_value)
            print(f"🔺 Joint{self.selected_joint+1}: {new_value:.3f} (정밀)")
        
        elif key == 'a':
            new_value = current_joints[self.selected_joint] - 0.01
            self.set_joint_value(self.selected_joint, new_value)
            print(f"🔻 Joint{self.selected_joint+1}: {new_value:.3f} (정밀)")
        
        # 데이터 수집 관련
        elif key == 'm':
            self.data_collection_mode = not self.data_collection_mode
            mode_str = "활성화" if self.data_collection_mode else "비활성화"
            print(f"🎯 데이터 수집 모드 {mode_str}")
        
        elif key == 'c':
            self.collect_mapping_data()
        
        elif key == 's':
            self.save_collected_data()
        
        elif key == 'l':
            self.load_existing_data()
        
        # 기타
        elif key == 'r':
            self.reset_robot_pose()
            print("🔄 로봇 리셋됨")
        
        elif key == ' ':
            joints = self.get_current_joint_values()
            vr_pos = self.get_vr_position_delta()
            print(f"📍 현재 조인트: [{', '.join([f'{j:.3f}' for j in joints])}]")
            print(f"📍 VR 위치: [{vr_pos[0]:+.3f}, {vr_pos[1]:+.3f}, {vr_pos[2]:+.3f}]")
        
        elif key == '\x1b':  # ESC
            print("🏁 종료")
            self.running = False
    
    def print_current_status(self):
        """현재 상태 출력"""
        joints = self.get_current_joint_values()
        vr_pos = self.get_vr_position_delta()
        print(f"🤖 조인트: J1={joints[0]:.3f}, J2={joints[1]:.3f}, J3={joints[2]:.3f}, J4={joints[3]:.3f}")
        print(f"🎯 선택: Joint{self.selected_joint+1}")
        print(f"📊 수집: {'활성' if self.data_collection_mode else '비활성'} | 총 {len(self.collected_mappings)}개")
        print(f"🖐 VR 위치: [{vr_pos[0]:+.3f}, {vr_pos[1]:+.3f}, {vr_pos[2]:+.3f}]")
    
    def run(self):
        """메인 실행 루프"""
        # 기존 데이터 로드 시도
        self.load_existing_data()
        
        # 키보드 입력 스레드 시작
        keyboard_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)
        keyboard_thread.start()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("\n🎯 매핑 데이터 수집기 시작!")
            print("=" * 60)
            print("⌨️ 키보드 조작:")
            print("  1-4: 조인트 선택")
            print("  +/-: 조인트 조작 (큰 단위 0.05)")
            print("  q/a: 조인트 조작 (작은 단위 0.01)")
            print("  r: 로봇 리셋")
            print("  SPACE: 현재 상태 출력")
            print("")
            print("📊 데이터 수집:")
            print("  m: 데이터 수집 모드 토글")
            print("  c: 현재 매핑 데이터 수집")
            print("  s: 수집된 데이터 저장")
            print("  l: 기존 데이터 로드")
            print("  ESC: 종료")
            print("=" * 60)
            print("🎯 사용법: 로봇을 원하는 자세로 조정 → VR 컨트롤러 맞춤 → 'c'로 수집")
            print("=" * 60)
            
            self.print_current_status()
            
            last_status_time = time.time()
            
            while viewer.is_running() and self.running:
                # 키보드 입력 처리
                self.handle_keyboard_input()
                
                # 물리 시뮬레이션
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 10초마다 상태 출력
                if time.time() - last_status_time > 10.0:
                    self.print_current_status()
                    last_status_time = time.time()
                
                time.sleep(0.01)
        
        self.running = False
        
        # 종료 시 자동 저장
        if self.collected_mappings:
            print("🔄 종료 전 데이터 자동 저장...")
            self.save_collected_data()
        
        print("🏁 매핑 데이터 수집기 종료")

if __name__ == "__main__":
    try:
        collector = MappingDataCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
