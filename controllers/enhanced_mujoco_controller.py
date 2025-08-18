#!/usr/bin/env python3
"""
🎯 Enhanced MuJoCo Controller with Real-time Data Collection
- VR Bridge와 양방향 통신
- 키보드로 로봇 수동 조정
- 실시간 매핑 데이터 수집
- 사용자 친화적 인터페이스
"""

import socket
import json
import time
import numpy as np
import mujoco
import mujoco.viewer
import threading
from collections import deque

class EnhancedMuJoCoController:
    def __init__(self):
        print("🎯 Enhanced MuJoCo Controller with Data Collection 초기화 중...")
        
        # 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 단일 로봇 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 scene.xml과 omx.xml이 같은 폴더에 있는지 확인하세요")
            raise
        
        # VR 브릿지 연결
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
        # 액추에이터 매핑
        self.joint_mapping = {}
        self.gripper_mapping = -1
        
        # 제어 모드
        self.control_modes = {
            'VR': 'vr_control',
            'MANUAL': 'manual_control',
            'COLLECTION': 'data_collection'
        }
        self.current_mode = self.control_modes['VR']
        
        # 수동 제어 상태
        self.manual_joints = [0.0, -0.3, 0.8, 0.0]  # 초기 안전 자세
        self.manual_gripper = -0.01
        self.selected_joint = 0
        
        # 데이터 수집 상태
        self.data_collection_mode = False
        self.pending_collection = False
        
        # VR 데이터
        self.vr_joints = [0.0, -0.3, 0.8, 0.0]
        self.vr_gripper = -0.01
        
        # 성능 모니터링
        self.performance_stats = {
            'fps': 0.0,
            'data_receive_rate': 0.0,
            'last_data_time': time.time(),
            'frame_times': deque(maxlen=60),
            'data_receive_times': deque(maxlen=60),
            'last_print_time': time.time(),
            'total_frames': 0,
            'successful_updates': 0
        }
        
        # VR 상태 추적
        self.vr_status = {
            'calibrated': False,
            'trigger_value': 0.0,
            'button_upper': False,
            'button_lower': False,
            'control_frequency': 0.0
        }
        
        # 수집 통계
        self.collection_stats = {
            'total_points': 0,
            'session_collected': 0
        }
        
        # 데이터 버퍼
        self.data_buffer = ""
        
        # 조인트 제한값
        self.joint_limits = {
            0: [-3.14, 3.14],    # joint1
            1: [-1.5, 1.5],     # joint2
            2: [-1.5, 1.4],     # joint3
            3: [-1.7, 1.97]     # joint4
        }
        
        # 키보드 상태
        self.keys_pressed = set()
        
        # 액추에이터 매핑 설정
        self.setup_actuator_mapping()
        
        # 안전한 초기 위치 설정
        self.reset_robot_pose()
        
        print("✅ Enhanced MuJoCo Controller 초기화 완료!")
    
    def setup_actuator_mapping(self):
        """액추에이터 매핑 설정"""
        print("🔧 액추에이터 매핑 설정 중...")
        
        actuator_patterns = [
            'actuator_joint1',
            'actuator_joint2', 
            'actuator_joint3',
            'actuator_joint4',
            'actuator_gripper_joint'
        ]
        
        for joint_idx, pattern in enumerate(actuator_patterns[:-1]):
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern)
                if actuator_id >= 0:
                    self.joint_mapping[joint_idx] = actuator_id
                    print(f"✅ Joint{joint_idx+1} → 액추에이터 {actuator_id}")
                else:
                    print(f"❌ Joint{joint_idx+1}: {pattern} 찾을 수 없음")
                    self.joint_mapping[joint_idx] = -1
            except Exception as e:
                print(f"❌ Joint{joint_idx+1} 매핑 오류: {e}")
                self.joint_mapping[joint_idx] = -1
        
        # 그리퍼 매핑
        try:
            gripper_pattern = 'actuator_gripper_joint'
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_pattern)
            if actuator_id >= 0:
                self.gripper_mapping = actuator_id
                print(f"✅ 그리퍼 → 액추에이터 {actuator_id}")
            else:
                self.gripper_mapping = -1
                print(f"❌ 그리퍼: {gripper_pattern} 찾을 수 없음")
        except Exception as e:
            print(f"❌ 그리퍼 매핑 오류: {e}")
            self.gripper_mapping = -1
    
    def reset_robot_pose(self):
        """로봇을 안전한 초기 자세로 설정"""
        print("🔄 로봇을 안전한 초기 자세로 설정 중...")
        
        for joint_idx, angle in enumerate(self.manual_joints):
            actuator_id = self.joint_mapping.get(joint_idx, -1)
            if actuator_id >= 0:
                safe_angle = np.clip(angle, 
                                   self.joint_limits[joint_idx][0], 
                                   self.joint_limits[joint_idx][1])
                self.data.ctrl[actuator_id] = safe_angle
        
        if self.gripper_mapping >= 0:
            self.data.ctrl[self.gripper_mapping] = self.manual_gripper
        
        # 물리 시뮬레이션으로 안정화
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        
        print("✅ 로봇 초기 자세 설정 완료")
    
    def connect_to_bridge(self):
        """VR 브릿지에 연결"""
        while self.connection_attempts < self.max_connection_attempts:
            try:
                print(f"🔗 VR 브릿지 연결 시도 {self.connection_attempts + 1}/{self.max_connection_attempts}...")
                self.socket.connect(('localhost', 12345))
                self.socket.settimeout(0.001)
                self.connected = True
                print("✅ VR 브릿지 연결 성공!")
                return True
                
            except Exception as e:
                self.connection_attempts += 1
                print(f"❌ 연결 실패: {e}")
                
                if self.connection_attempts < self.max_connection_attempts:
                    print(f"⏳ 2초 후 재시도...")
                    time.sleep(2.0)
                else:
                    print("❌ 최대 연결 시도 횟수 초과")
        
        print("⚠️ VR 브릿지 없이 수동 제어 모드로 시작합니다")
        self.current_mode = self.control_modes['MANUAL']
        return False
    
    def receive_vr_data(self):
        """VR 데이터 수신"""
        if not self.connected:
            return None
        
        try:
            raw_data = self.socket.recv(8192).decode('utf-8', errors='ignore')
            if not raw_data:
                return None
            
            self.data_buffer += raw_data
            
            while '\n' in self.data_buffer:
                line, self.data_buffer = self.data_buffer.split('\n', 1)
                line = line.strip()
                
                if line:
                    try:
                        parsed_data = json.loads(line)
                        
                        # 데이터 수신 시간 기록
                        current_time = time.time()
                        if len(self.performance_stats['data_receive_times']) > 0:
                            dt = current_time - self.performance_stats['last_data_time']
                            if dt > 0:
                                self.performance_stats['data_receive_times'].append(dt)
                        self.performance_stats['last_data_time'] = current_time
                        
                        # VR 상태 업데이트
                        if 'vr_status' in parsed_data:
                            self.vr_status.update(parsed_data['vr_status'])
                        
                        # 수집 상태 업데이트
                        if 'collection_status' in parsed_data:
                            collection_status = parsed_data['collection_status']
                            self.data_collection_mode = collection_status.get('mode_active', False)
                            self.collection_stats['total_points'] = collection_status.get('total_points', 0)
                            self.collection_stats['session_collected'] = collection_status.get('session_collected', 0)
                        
                        # 조인트 데이터 업데이트
                        if 'joint_angles' in parsed_data and 'gripper' in parsed_data:
                            self.vr_joints = parsed_data['joint_angles'][:4]
                            self.vr_gripper = parsed_data['gripper']
                        
                        return parsed_data
                        
                    except json.JSONDecodeError:
                        continue
                        
        except socket.timeout:
            pass
        except Exception as e:
            print(f"⚠️ 데이터 수신 오류: {e}")
            self.connected = False
        
        return None
    
    def send_to_bridge(self, command_data):
        """VR 브릿지로 명령 전송"""
        if not self.connected:
            return
        
        try:
            message = json.dumps(command_data) + '\n'
            self.socket.sendall(message.encode())
        except Exception as e:
            print(f"⚠️ 브릿지 전송 오류: {e}")
            self.connected = False
    
    def update_robot_from_mode(self):
        """현재 모드에 따라 로봇 업데이트"""
        if self.current_mode == self.control_modes['VR']:
            # VR 제어 모드
            target_joints = self.vr_joints
            target_gripper = self.vr_gripper
            
        elif self.current_mode == self.control_modes['MANUAL'] or self.current_mode == self.control_modes['COLLECTION']:
            # 수동 제어 모드 또는 데이터 수집 모드
            target_joints = self.manual_joints
            target_gripper = self.manual_gripper
            
        else:
            return
        
        try:
            # 조인트 업데이트
            for joint_idx, angle in enumerate(target_joints[:4]):
                actuator_id = self.joint_mapping.get(joint_idx, -1)
                if actuator_id >= 0:
                    safe_angle = np.clip(angle, 
                                       self.joint_limits[joint_idx][0], 
                                       self.joint_limits[joint_idx][1])
                    
                    if not np.isnan(safe_angle) and not np.isinf(safe_angle):
                        self.data.ctrl[actuator_id] = safe_angle
            
            # 그리퍼 업데이트
            if self.gripper_mapping >= 0:
                safe_gripper = np.clip(target_gripper, -0.01, 0.019)
                if not np.isnan(safe_gripper) and not np.isinf(safe_gripper):
                    self.data.ctrl[self.gripper_mapping] = safe_gripper
            
            self.performance_stats['successful_updates'] += 1
            
        except Exception as e:
            print(f"⚠️ 로봇 업데이트 오류: {e}")
    
    def handle_keyboard_input(self, window, key, scancode, action, mods):
        """키보드 입력 처리"""
        if action == 1:  # 키 누름
            self.keys_pressed.add(key)
        elif action == 0:  # 키 떼기
            self.keys_pressed.discard(key)
        
        if action == 1:  # 키 누름 이벤트만 처리
            
            # 모드 전환
            if key == 86:  # 'V' - VR 모드
                if self.connected:
                    self.current_mode = self.control_modes['VR']
                    print("🎮 VR 제어 모드")
                else:
                    print("❌ VR 브릿지가 연결되지 않음")
                    
            elif key == 77:  # 'M' - 수동 모드
                self.current_mode = self.control_modes['MANUAL']
                print("⌨️ 수동 제어 모드")
                
            elif key == 67:  # 'C' - 데이터 수집 모드
                self.current_mode = self.control_modes['COLLECTION']
                self.data_collection_mode = not self.data_collection_mode
                
                # VR 브릿지에 모드 변경 알림
                self.send_to_bridge({
                    'command': 'data_collection_mode',
                    'enabled': self.data_collection_mode
                })
                
                mode_str = "활성화" if self.data_collection_mode else "비활성화"
                print(f"🎯 데이터 수집 모드 {mode_str}")
            
            # 조인트 선택 (숫자 키 1-4)
            elif 49 <= key <= 52:  # '1'-'4'
                self.selected_joint = key - 49
                print(f"🎯 Joint {self.selected_joint + 1} 선택됨")
            
            # 조인트 제어
            elif key == 265:  # UP 화살표
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.adjust_joint(self.selected_joint, +0.1)
                    
            elif key == 264:  # DOWN 화살표
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.adjust_joint(self.selected_joint, -0.1)
            
            # 그리퍼 제어
            elif key == 79:  # 'O' - 그리퍼 열기
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.manual_gripper = 0.019
                    print("🤏 그리퍼 열기")
                    
            elif key == 80:  # 'P' - 그리퍼 닫기
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.manual_gripper = -0.01
                    print("✊ 그리퍼 닫기")
            
            # 데이터 수집
            elif key == 32:  # SPACE - 현재 매핑 저장
                if self.current_mode == self.control_modes['COLLECTION'] and self.data_collection_mode:
                    self.collect_current_mapping()
            
            # 리셋
            elif key == 82:  # 'R' - 리셋
                self.manual_joints = [0.0, -0.3, 0.8, 0.0]
                self.manual_gripper = -0.01
                print("🔄 수동 조인트 리셋")
            
            # 저장
            elif key == 83:  # 'S' - 저장
                self.send_to_bridge({'command': 'save_data'})
                print("💾 매핑 데이터 저장 요청")
    
    def adjust_joint(self, joint_idx, delta):
        """조인트 값 조정"""
        if 0 <= joint_idx < 4:
            self.manual_joints[joint_idx] += delta
            
            # 제한 적용
            limits = self.joint_limits[joint_idx]
            self.manual_joints[joint_idx] = np.clip(
                self.manual_joints[joint_idx], limits[0], limits[1]
            )
            
            print(f"🔧 Joint{joint_idx+1}: {self.manual_joints[joint_idx]:.2f}")
            
            # VR 브릿지에 현재 조인트 상태 전송
            self.send_to_bridge({
                'command': 'set_robot_joints',
                'joints': self.manual_joints
            })
    
    def collect_current_mapping(self):
        """현재 위치를 매핑 데이터로 수집"""
        # VR 브릿지에 수집 명령 전송
        self.send_to_bridge({
            'command': 'collect_data'
        })
        
        print("📊 매핑 데이터 수집 요청 전송")
    
    def update_performance_stats(self, frame_time):
        """성능 통계 업데이트"""
        self.performance_stats['frame_times'].append(frame_time)
        self.performance_stats['total_frames'] += 1
        
        if len(self.performance_stats['frame_times']) > 0:
            avg_frame_time = np.mean(self.performance_stats['frame_times'])
            self.performance_stats['fps'] = 1.0 / max(avg_frame_time, 0.001)
        
        if len(self.performance_stats['data_receive_times']) > 0:
            avg_receive_time = np.mean(self.performance_stats['data_receive_times'])
            self.performance_stats['data_receive_rate'] = 1.0 / max(avg_receive_time, 0.001)
    
    def print_status(self):
        """상태 정보 출력"""
        current_time = time.time()
        if current_time - self.performance_stats['last_print_time'] > 3.0:
            
            mode_names = {
                'vr_control': 'VR 제어',
                'manual_control': '수동 제어', 
                'data_collection': '데이터 수집'
            }
            mode_name = mode_names.get(self.current_mode, '알 수 없음')
            
            print(f"\n🎯 === Enhanced MuJoCo Controller ===")
            print(f"🎮 제어 모드: {mode_name}")
            print(f"🖐 VR 캘리브레이션: {'✅' if self.vr_status['calibrated'] else '❌'}")
            print(f"🔗 VR 연결: {'✅' if self.connected else '❌'}")
            print(f"📊 FPS: {self.performance_stats['fps']:.1f}")
            print(f"🎯 수집 모드: {'✅ 활성' if self.data_collection_mode else '❌ 비활성'}")
            print(f"📈 총 매핑 포인트: {self.collection_stats['total_points']}개")
            print(f"📍 이번 세션: {self.collection_stats['session_collected']}개")
            
            if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                print(f"🤖 수동 조인트: {[f'{j:.2f}' for j in self.manual_joints]}")
                print(f"🎯 선택된 조인트: Joint{self.selected_joint + 1}")
            else:
                print(f"🤖 VR 조인트: {[f'{j:.2f}' for j in self.vr_joints]}")
            
            print("-" * 50)
            
            self.performance_stats['last_print_time'] = current_time
            self.performance_stats['successful_updates'] = 0
    
    def run(self):
        """메인 실행 루프"""
        # VR 브릿지 연결 시도
        self.connect_to_bridge()
        
        # MuJoCo 뷰어 시작
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                        key_callback=self.handle_keyboard_input) as viewer:
            
            # 카메라 설정
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.2, 0, 0.2]
            
            print("\n🎯 Enhanced MuJoCo Controller 시작!")
            print("=" * 60)
            print("⌨️ 제어 명령어:")
            print("   V: VR 제어 모드")
            print("   M: 수동 제어 모드") 
            print("   C: 데이터 수집 모드 토글")
            print("   1-4: 조인트 선택")
            print("   ↑/↓: 선택된 조인트 제어")
            print("   O: 그리퍼 열기")
            print("   P: 그리퍼 닫기")
            print("   SPACE: 현재 매핑 저장 (수집 모드)")
            print("   R: 수동 조인트 리셋")
            print("   S: 매핑 데이터 저장")
            print("=" * 60)
            print("🎯 데이터 수집 방법:")
            print("   1. C키로 수집 모드 활성화")
            print("   2. 키보드로 로봇을 원하는 자세로 조정")
            print("   3. VR 컨트롤러를 해당 위치로 이동")
            print("   4. SPACE키로 매핑 저장")
            print("=" * 60)
            
            frame_count = 0
            
            while viewer.is_running():
                frame_start_time = time.time()
                
                # VR 데이터 수신
                vr_data = self.receive_vr_data()
                
                # 로봇 업데이트
                self.update_robot_from_mode()
                
                # 물리 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                
                # 뷰어 동기화
                viewer.sync()
                
                # 성능 통계 업데이트
                frame_time = time.time() - frame_start_time
                self.update_performance_stats(frame_time)
                
                # 상태 출력 (3초마다)
                if frame_count % 300 == 0:
                    self.print_status()
                
                frame_count += 1
                
                # 프레임 레이트 제한
                time.sleep(max(0, 0.008 - frame_time))
        
        print("🏁 Enhanced MuJoCo Controller 종료")
    
    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'socket') and self.connected:
                self.socket.close()
            print("🧹 리소스 정리 완료")
        except:
            pass

if __name__ == "__main__":
    try:
        print("🎯 Enhanced MuJoCo Controller with Data Collection 시작...")
        controller = EnhancedMuJoCoController()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 시스템 종료")
