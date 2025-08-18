#!/usr/bin/env python3
"""
🎯 MuJoCo 키보드 제어 + VR 데이터 수집 시스템
- 키보드로 로봇 조인트 정밀 제어
- VR 컨트롤러 위치와 실시간 매칭
- 원하는 순간에 데이터 수집
- 200개+ 고품질 매핑 데이터 생성
"""

import mujoco
import mujoco.viewer
import numpy as np
import json
import socket
import threading
import time
from datetime import datetime
from collections import deque

class MuJoCoKeyboardDataCollector:
    def __init__(self):
        print("🎯 MuJoCo 키보드 제어 + 데이터 수집 시스템 초기화...")
        
        # MuJoCo 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # VR 브릿지 연결 (데이터 수집용)
        self.setup_vr_connection()
        
        # 로봇 제어 상태
        self.current_joints = [0.0, -0.3, 0.8, 0.0]  # 초기 안전 자세
        self.selected_joint = 0  # 현재 선택된 조인트 (0-3)
        self.joint_step = 0.05   # 조인트 증감 스텝
        self.fine_mode = False   # 정밀 제어 모드
        
        # 조인트 제한
        self.joint_limits = [
            [-3.14, 3.14],  # joint1
            [-1.5, 1.5],    # joint2
            [-1.5, 1.4],    # joint3
            [-2.0, 2.0]     # joint4
        ]
        
        # VR 데이터
        self.vr_data = {
            'calibrated': False
        }
        
        # 현재 VR 델타 (실시간 업데이트)
        self.current_vr_deltas = {
            'position_delta': [0.0, 0.0, 0.0],
            'orientation_delta': [0.0, 0.0, 0.0]
        }
        
        # 수집된 데이터
        self.collected_samples = []
        self.target_samples = 200
        
        # 액추에이터 매핑
        self.joint_mapping = {}
        self.setup_actuator_mapping()
        
        # 초기 자세 설정
        self.reset_robot()
        
        print("✅ 키보드 제어 + 데이터 수집 시스템 준비 완료!")
        print("\n🎮 === 키보드 조작법 ===")
        print("조인트 선택:")
        print("  1,2,3,4: Joint 1~4 선택")
        print("조인트 제어:")  
        print("  Q/A: 선택된 조인트 +/- (큰 스텝)")
        print("  W/S: 선택된 조인트 +/- (작은 스텝)")
        print("  F: 정밀모드 토글 (0.01 스텝)")
        print("데이터 수집:")
        print("  SPACE: 현재 위치에서 데이터 수집")
        print("  C: VR 컨트롤러 재캘리브레이션")
        print("  R: 로봇 리셋 (안전 자세)")
        print("  P: 수집된 데이터 저장 (JSON)")
        print("  ESC: 종료")
    
    def setup_vr_connection(self):
        """VR 브릿지 연결 (수신 전용)"""
        try:
            self.vr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.vr_socket.connect(('localhost', 12346))  # 포트 수정!
            self.vr_socket.settimeout(0.01)  # 논블로킹
            self.vr_connected = True
            print("✅ VR 브릿지 연결 완료")
            
            # VR 데이터 수신 스레드
            self.vr_thread = threading.Thread(target=self.vr_receive_loop, daemon=True)
            self.vr_thread.start()
            
        except Exception as e:
            print(f"⚠️ VR 브릿지 연결 실패: {e}")
            print("VR 없이 로봇 제어만 가능합니다")
            self.vr_connected = False
    
    def vr_receive_loop(self):
        """VR 데이터 수신 루프"""
        buffer = ""
        
        while self.vr_connected:
            try:
                data = self.vr_socket.recv(1024).decode('utf-8')
                if not data:
                    continue
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            vr_msg = json.loads(line.strip())
                            self.process_vr_data(vr_msg)
                        except json.JSONDecodeError:
                            pass
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"VR 연결 오류: {e}")
                self.vr_connected = False
                break
    
    def process_vr_data(self, vr_msg):
        """VR 데이터 처리"""
        try:
            # VR 상태 업데이트
            if 'vr_status' in vr_msg:
                self.vr_data['calibrated'] = vr_msg['vr_status'].get('calibrated', False)
            
            # VR 델타 데이터 저장
            if 'vr_deltas' in vr_msg:
                vr_deltas = vr_msg['vr_deltas']
                self.current_vr_deltas = {
                    'position_delta': vr_deltas.get('position_delta', [0, 0, 0]),
                    'orientation_delta': vr_deltas.get('orientation_delta', [0, 0, 0])
                }
                
        except Exception as e:
            print(f"VR 데이터 처리 오류: {e}")
    
    def setup_actuator_mapping(self):
        """액추에이터 매핑 설정"""
        patterns = ['actuator_joint1', 'actuator_joint2', 'actuator_joint3', 'actuator_joint4']
        
        for i, pattern in enumerate(patterns):
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern)
                if actuator_id >= 0:
                    self.joint_mapping[i] = actuator_id
                    print(f"✅ Joint{i+1} → 액추에이터 {actuator_id}")
                else:
                    print(f"❌ {pattern} 찾을 수 없음")
            except Exception as e:
                print(f"❌ Joint{i+1} 매핑 오류: {e}")
    
    def reset_robot(self):
        """로봇을 안전한 초기 자세로 리셋"""
        self.current_joints = [0.0, -0.3, 0.8, 0.0]
        self.update_robot_joints()
        print("🔄 로봇 안전 자세로 리셋")
    
    def update_robot_joints(self):
        """현재 조인트 각도를 MuJoCo에 적용"""
        for i, angle in enumerate(self.current_joints):
            actuator_id = self.joint_mapping.get(i, -1)
            if actuator_id >= 0:
                # 안전 범위 클리핑
                safe_angle = np.clip(angle, self.joint_limits[i][0], self.joint_limits[i][1])
                self.data.ctrl[actuator_id] = safe_angle
    
    def get_end_effector_pose(self):
        """End-effector의 현재 위치와 자세"""
        try:
            # end_effector_target body 찾기
            ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_target')
            if ee_body_id >= 0:
                pos = self.data.xpos[ee_body_id].copy()
                quat = self.data.xquat[ee_body_id].copy()  # [w, x, y, z]
                return pos, quat
            else:
                # 대안: link5 사용
                link5_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link5')
                pos = self.data.xpos[link5_id].copy()
                quat = self.data.xquat[link5_id].copy()
                return pos, quat
        except Exception as e:
            print(f"End-effector pose 획득 실패: {e}")
            return np.array([0.2, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])
    
    def collect_data_sample(self):
        """현재 상태에서 데이터 샘플 수집"""
        if not self.vr_data['calibrated']:
            print("❌ VR 컨트롤러가 캘리브레이션되지 않았습니다!")
            print("VR 브릿지에서 A+B 버튼을 눌러 캘리브레이션하세요")
            return False
        
        # 현재 로봇 상태
        ee_pos, ee_quat = self.get_end_effector_pose()
        
        # 실시간 VR 델타 사용
        vr_pos_delta = self.current_vr_deltas['position_delta']
        vr_ori_delta = self.current_vr_deltas['orientation_delta']
        
        # 데이터 샘플 생성
        sample = {
            'sample_id': len(self.collected_samples) + 1,
            'timestamp': time.time(),
            
            # VR 데이터 (실제 브릿지에서 받은 값)
            'vr_position_delta': vr_pos_delta,
            'vr_orientation_delta': vr_ori_delta,
            
            # 로봇 데이터  
            'joint_angles': self.current_joints.copy(),
            'end_effector_position': ee_pos.tolist(),
            'end_effector_quaternion': ee_quat.tolist(),
            
            # 수집 정보
            'collection_method': 'manual_keyboard',
            'selected_joint_at_collection': self.selected_joint,
            'fine_mode_active': self.fine_mode
        }
        
        # 유효성 검증
        if self.is_valid_sample(sample):
            self.collected_samples.append(sample)
            print(f"✅ 샘플 {len(self.collected_samples)}/{self.target_samples} 수집 완료!")
            print(f"   VR Delta: X={vr_pos_delta[0]:+.3f}, Y={vr_pos_delta[1]:+.3f}, Z={vr_pos_delta[2]:+.3f}")
            print(f"   End-Effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"   Joints: [{self.current_joints[0]:.2f}, {self.current_joints[1]:.2f}, {self.current_joints[2]:.2f}, {self.current_joints[3]:.2f}]")
            
            # 진행률 표시
            progress = (len(self.collected_samples) / self.target_samples) * 100
            print(f"   진행률: {progress:.1f}%")
            
            return True
        else:
            print("❌ 유효하지 않은 샘플 (범위 벗어남 또는 중복)")
            return False
    
    def is_valid_sample(self, sample):
        """샘플 유효성 검증"""
        # 1. 작업공간 범위 체크
        ee_pos = sample['end_effector_position']
        if (ee_pos[0] < 0.05 or ee_pos[0] > 0.4 or 
            abs(ee_pos[1]) > 0.3 or 
            ee_pos[2] < 0.05 or ee_pos[2] > 0.5):
            return False
        
        # 2. 조인트 범위 체크
        for i, angle in enumerate(sample['joint_angles']):
            if (angle < self.joint_limits[i][0] or 
                angle > self.joint_limits[i][1]):
                return False
        
        # 3. 중복 방지 (최소 거리)
        min_distance = 0.03  # 3cm 최소 거리
        for existing in self.collected_samples:
            existing_pos = existing['end_effector_position']
            distance = np.linalg.norm(np.array(ee_pos) - np.array(existing_pos))
            if distance < min_distance:
                return False
        
        return True
    
    def save_collected_data(self):
        """수집된 데이터 저장 (개수 제한 없음)"""
        if not self.collected_samples:
            print("❌ 저장할 데이터가 없습니다")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keyboard_collected_mapping_{timestamp}.json"
        
        # 메타데이터와 함께 저장
        data_package = {
            'metadata': {
                'collection_method': 'manual_keyboard_control',
                'total_samples': len(self.collected_samples),
                'target_samples': self.target_samples,
                'completion_rate': f"{(len(self.collected_samples)/self.target_samples)*100:.1f}%",
                'collection_timestamp': timestamp,
                'joint_limits': self.joint_limits,
                'collection_status': 'completed' if len(self.collected_samples) >= self.target_samples else 'partial'
            },
            'samples': self.collected_samples
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data_package, f, indent=2)
            
            print(f"✅ 데이터 저장 완료!")
            print(f"   파일명: {filename}")
            print(f"   수집된 샘플: {len(self.collected_samples)}개")
            print(f"   목표 대비: {(len(self.collected_samples)/self.target_samples)*100:.1f}%")
            
            if len(self.collected_samples) < self.target_samples:
                print(f"   💡 목표({self.target_samples}개)보다 적지만 유효한 데이터셋입니다!")
            
            # 간단한 통계
            self.print_collection_stats()
            
            return filename
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return None
    
    def print_collection_stats(self):
        """수집 통계 출력"""
        if not self.collected_samples:
            return
        
        positions = np.array([s['end_effector_position'] for s in self.collected_samples])
        
        print("\n📊 === 수집 데이터 통계 ===")
        print(f"총 샘플 수: {len(self.collected_samples)}")
        print(f"X 범위: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print(f"Y 범위: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print(f"Z 범위: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        # 각 조인트별 사용 범위
        joints = np.array([s['joint_angles'] for s in self.collected_samples])
        for i in range(4):
            print(f"Joint{i+1} 범위: [{joints[:, i].min():.3f}, {joints[:, i].max():.3f}]")
    
    def keyboard_handler(self, keycode):
        """키보드 입력 처리"""
        key = chr(keycode).upper() if 32 <= keycode <= 126 else None
        
        if key:
            # 조인트 선택
            if key in '1234':
                self.selected_joint = int(key) - 1
                print(f"🎯 Joint{self.selected_joint + 1} 선택됨")
                
            # 조인트 제어 (큰 스텝)
            elif key == 'Q':
                step = 0.01 if self.fine_mode else self.joint_step
                self.current_joints[self.selected_joint] += step
                self.update_robot_joints()
                print(f"Joint{self.selected_joint + 1}: {self.current_joints[self.selected_joint]:.3f}")
                
            elif key == 'A':
                step = 0.01 if self.fine_mode else self.joint_step
                self.current_joints[self.selected_joint] -= step
                self.update_robot_joints()
                print(f"Joint{self.selected_joint + 1}: {self.current_joints[self.selected_joint]:.3f}")
                
            # 조인트 제어 (작은 스텝)
            elif key == 'W':
                step = 0.005 if self.fine_mode else 0.01
                self.current_joints[self.selected_joint] += step
                self.update_robot_joints()
                print(f"Joint{self.selected_joint + 1}: {self.current_joints[self.selected_joint]:.3f}")
                
            elif key == 'S':
                step = 0.005 if self.fine_mode else 0.01
                self.current_joints[self.selected_joint] -= step
                self.update_robot_joints()
                print(f"Joint{self.selected_joint + 1}: {self.current_joints[self.selected_joint]:.3f}")
                
            # 정밀 모드 토글
            elif key == 'F':
                self.fine_mode = not self.fine_mode
                mode_text = "정밀" if self.fine_mode else "일반"
                print(f"🔧 {mode_text} 제어 모드")
                
            # 데이터 수집
            elif key == ' ':  # SPACE
                self.collect_data_sample()
                
            # VR 재캘리브레이션
            elif key == 'C':
                print("🔄 VR 컨트롤러 재캘리브레이션 요청")
                print("VR 브릿지에서 A+B 버튼을 동시에 누르세요")
                
            # 로봇 리셋
            elif key == 'R':
                self.reset_robot()
                
            # 데이터 저장 (P 키)
            elif key == 'P':
                self.save_collected_data()
    
    def run(self):
        """메인 실행 루프"""
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_handler) as viewer:
            
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.lookat = [0.2, 0, 0.2]
            
            print("\n🚀 키보드 제어 + 데이터 수집 시스템 시작!")
            print("=" * 60)
            print("🎯 목표: 200개 고품질 VR-Robot 매핑 데이터 수집")
            print("🎮 키보드로 로봇을 원하는 위치로 이동")
            print("🖐 VR 컨트롤러를 같은 위치에 맞춤")
            print("⌨️ SPACE로 데이터 수집!")
            print("💾 P 키로 JSON 저장!")
            print("=" * 60)
            
            frame_count = 0
            
            while viewer.is_running():
                # 물리 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                
                # 뷰어 동기화
                viewer.sync()
                
                # 상태 표시 (5초마다)
                if frame_count % 600 == 0:
                    ee_pos, _ = self.get_end_effector_pose()
                    print(f"\n📊 현재 상태:")
                    print(f"   선택된 조인트: Joint{self.selected_joint + 1}")
                    print(f"   제어 모드: {'정밀' if self.fine_mode else '일반'}")
                    print(f"   End-Effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]") 
                    print(f"   수집된 샘플: {len(self.collected_samples)}/{self.target_samples}")
                    print(f"   진행률: {(len(self.collected_samples)/self.target_samples)*100:.1f}%")
                
                frame_count += 1
                time.sleep(0.01)  # 100Hz
        
        print("🏁 키보드 제어 + 데이터 수집 시스템 종료")

if __name__ == "__main__":
    try:
        collector = MuJoCoKeyboardDataCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 시스템 종료")

