#!/usr/bin/env python3
"""
🎯 MuJoCo Single Robot Controller (Host용)
- scene.xml 로드 (omx.xml 포함)
- VR Bridge로부터 데이터 수신
- 정밀한 단일 로봇 제어
- 실시간 성능 모니터링
"""

import socket
import json
import time
import numpy as np
import mujoco
import mujoco.viewer
import threading
from collections import deque

class PrecisionMuJoCoController:
    def __init__(self):
        print("🎯 정밀 MuJoCo 단일 로봇 컨트롤러 초기화 중...")
        
        # 모델 로드 (scene.xml → omx.xml 포함)
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
        
        # 액추에이터 매핑 (단일 로봇용)
        self.joint_mapping = {}
        self.gripper_mapping = -1
        
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
            'ik_success_rate': 0.0,
            'control_frequency': 0.0
        }
        
        # 데이터 버퍼
        self.data_buffer = ""
        self.invalid_data_count = 0
        
        # 조인트 제한값 (안전성)
        self.joint_limits = {
            'joint1': [-3.14, 3.14],
            'joint2': [-1.5, 1.5],
            'joint3': [-1.5, 1.4],
            'joint4': [-1.7, 1.97]
        }
        
        # 액추에이터 매핑 설정
        self.setup_actuator_mapping()
        
        # 안전한 초기 위치 설정
        self.reset_robot_pose()
        
        print("✅ 정밀 MuJoCo 컨트롤러 초기화 완료!")
    
    def setup_actuator_mapping(self):
        """단일 로봇 액추에이터 매핑 설정"""
        print("🔧 단일 로봇 액추에이터 매핑 설정 중...")
        
        # 액추에이터 목록 출력
        print(f"\n📋 총 {self.model.nu}개 액추에이터:")
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i:2d}: {name}")
        
        # 단일 로봇 액추에이터 매핑
        actuator_patterns = [
            'actuator_joint1',
            'actuator_joint2',
            'actuator_joint3',
            'actuator_joint4',
            'actuator_gripper_joint'
        ]
        
        for joint_idx, pattern in enumerate(actuator_patterns[:-1]):  # Joint 1~4
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pattern)
                if actuator_id >= 0:
                    self.joint_mapping[joint_idx] = actuator_id
                    print(f"✅ Joint{joint_idx+1} → 액추에이터 {actuator_id} ({pattern})")
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
                print(f"✅ 그리퍼 → 액추에이터 {actuator_id} ({gripper_pattern})")
            else:
                self.gripper_mapping = -1
                print(f"❌ 그리퍼: {gripper_pattern} 찾을 수 없음")
        except Exception as e:
            print(f"❌ 그리퍼 매핑 오류: {e}")
            self.gripper_mapping = -1
        
        print("🔧 단일 로봇 액추에이터 매핑 완료\n")
    
    def reset_robot_pose(self):
        """로봇을 안전한 초기 자세로 설정"""
        print("🔄 로봇을 안전한 초기 자세로 설정 중...")
        
        # 안전한 초기 자세 (직관적 제어용)
        initial_pose = [0.0, -0.3, 0.8, 0.0]  # Joint 1~4
        
        # 조인트 설정
        for joint_idx, angle in enumerate(initial_pose):
            actuator_id = self.joint_mapping.get(joint_idx, -1)
            if actuator_id >= 0:
                # 안전 범위 체크
                joint_name = f'joint{joint_idx+1}'
                if joint_name in self.joint_limits:
                    safe_angle = np.clip(angle, 
                                       self.joint_limits[joint_name][0], 
                                       self.joint_limits[joint_name][1])
                else:
                    safe_angle = angle
                
                self.data.ctrl[actuator_id] = safe_angle
        
        # 그리퍼 초기화 (닫힌 상태)
        if self.gripper_mapping >= 0:
            self.data.ctrl[self.gripper_mapping] = -0.01
        
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
                self.socket.settimeout(0.001)  # 1ms 타임아웃
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
        
        print("⚠️ VR 브릿지 없이 시뮬레이션을 시작합니다")
        return False
    
    def receive_data(self):
        """VR 데이터 수신"""
        if not self.connected:
            return None
        
        try:
            # 데이터 수신
            raw_data = self.socket.recv(8192).decode('utf-8', errors='ignore')
            if not raw_data:
                return None
            
            # 버퍼에 추가
            self.data_buffer += raw_data
            
            # 완전한 JSON 메시지 파싱
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
                        
                        # 성능 데이터 업데이트
                        if 'performance' in parsed_data:
                            perf = parsed_data['performance']
                            if 'control_frequency' in perf:
                                self.vr_status['control_frequency'] = perf['control_frequency']
                        
                        return parsed_data
                        
                    except json.JSONDecodeError:
                        self.invalid_data_count += 1
                        continue
                        
        except socket.timeout:
            # 타임아웃은 정상 (논블로킹)
            pass
        except Exception as e:
            print(f"⚠️ 데이터 수신 오류: {e}")
            self.connected = False
        
        return None
    
    def update_robot(self, vr_data):
        """로봇 업데이트"""
        if not vr_data:
            return
        
        try:
            update_success = False
            
            # 조인트 각도 업데이트
            if 'joint_angles' in vr_data:
                joints = vr_data['joint_angles']
                
                for joint_idx, angle in enumerate(joints[:4]):  # Joint 1~4
                    actuator_id = self.joint_mapping.get(joint_idx, -1)
                    if actuator_id >= 0:
                        # 안전 범위 체크
                        joint_name = f'joint{joint_idx+1}'
                        if joint_name in self.joint_limits:
                            limits = self.joint_limits[joint_name]
                            safe_angle = np.clip(angle, limits[0], limits[1])
                        else:
                            safe_angle = np.clip(angle, -3.14, 3.14)
                        
                        # NaN/Inf 체크
                        if not np.isnan(safe_angle) and not np.isinf(safe_angle):
                            self.data.ctrl[actuator_id] = safe_angle
                            update_success = True
            
            # 그리퍼 업데이트
            if 'gripper' in vr_data and self.gripper_mapping >= 0:
                gripper_value = vr_data['gripper']
                safe_gripper = np.clip(gripper_value, -0.01, 0.019)
                
                if not np.isnan(safe_gripper) and not np.isinf(safe_gripper):
                    self.data.ctrl[self.gripper_mapping] = safe_gripper
            
            if update_success:
                self.performance_stats['successful_updates'] += 1
            
        except Exception as e:
            print(f"⚠️ 로봇 업데이트 오류: {e}")
    
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
        if current_time - self.performance_stats['last_print_time'] > 3.0:  # 3초마다
            
            # 성능 정보
            perf_info = (
                f"📊 FPS: {self.performance_stats['fps']:.1f} | "
                f"VR 데이터: {self.performance_stats['data_receive_rate']:.1f}Hz | "
                f"업데이트: {self.performance_stats['successful_updates']}"
            )
            
            # VR 상태 정보
            vr_info = (
                f"🖐 캘리브레이션: {'✅' if self.vr_status['calibrated'] else '❌'} | "
                f"🎮 트리거: {self.vr_status['trigger_value']:.2f}"
            )
            
            # 버튼 상태
            button_info = (
                f"🅰️ A: {'ON' if self.vr_status['button_upper'] else 'OFF'} | "
                f"🅱️ B: {'ON' if self.vr_status['button_lower'] else 'OFF'} | "
                f"Bridge: {self.vr_status['control_frequency']:.1f}Hz"
            )
            
            # 연결 상태
            connection_info = f"🌐 {'✅ VR 연결됨' if self.connected else '❌ VR 연결 끊김'}"
            
            # 현재 조인트 상태
            joint_info = "🤖 조인트: "
            for i in range(4):
                actuator_id = self.joint_mapping.get(i, -1)
                if actuator_id >= 0:
                    joint_info += f"J{i+1}={self.data.ctrl[actuator_id]:.2f} "
            
            if self.gripper_mapping >= 0:
                joint_info += f"G={self.data.ctrl[self.gripper_mapping]:.3f}"
            
            print(perf_info)
            print(vr_info)
            print(button_info)
            print(connection_info)
            print(joint_info)
            print("-" * 70)
            
            self.performance_stats['last_print_time'] = current_time
            self.performance_stats['successful_updates'] = 0
    
    def run(self):
        """메인 실행 루프"""
        # VR 브릿지 연결 시도
        self.connect_to_bridge()
        
        # MuJoCo 뷰어 시작
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 최적화된 카메라 설정 (단일 로봇용)
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.2, 0, 0.2]
            
            print("\n🎯 정밀 MuJoCo 단일 로봇 시뮬레이션 시작!")
            print("=" * 70)
            print("⌨️  조작 방법:")
            print("   마우스: 카메라 제어 (드래그: 회전, 휠: 줌)")
            print("   SPACE: 일시정지/재개")
            print("   ESC: 종료")
            print("   TAB: 정보 패널 토글")
            print("=" * 70)
            print("🖐 VR 컨트롤러로 로봇을 제어하세요!")
            print("🎮 A+B 버튼으로 VR 캘리브레이션")
            print("=" * 70)
            
            frame_count = 0
            
            while viewer.is_running():
                frame_start_time = time.time()
                
                # VR 데이터 수신 및 로봇 업데이트
                vr_data = self.receive_data()
                self.update_robot(vr_data)
                
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
                
                # 프레임 레이트 제한 (최대 120Hz)
                time.sleep(max(0, 0.008 - frame_time))
        
        print("🏁 정밀 MuJoCo 단일 로봇 컨트롤러 종료")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            if hasattr(self, 'socket') and self.connected:
                self.socket.close()
            print("🧹 리소스 정리 완료")
        except:
            pass

if __name__ == "__main__":
    try:
        print("🎯 정밀 MuJoCo 단일 로봇 시스템 시작...")
        controller = PrecisionMuJoCoController()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 시스템 종료")
