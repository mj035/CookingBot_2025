#!/usr/bin/env python3
"""
🎯 GLFW 기반 데이터 수집 MuJoCo 컨트롤러
- 직접 GLFW 키보드 처리로 안정적 입력
- VR Bridge와 양방향 통신
- 실시간 매핑 데이터 수집
- 키보드로 정밀한 로봇 제어
"""

import numpy as np
import mujoco
import time
import os
import socket
import json
import threading
from collections import deque
from mujoco.glfw import glfw

class GLFWDataCollector:
    def __init__(self):
        print("🎯 GLFW 데이터 수집 컨트롤러 초기화 중...")
        
        # MuJoCo 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # VR 브릿지 연결
        self.socket = None
        self.connected = False
        self.data_buffer = ""
        
        # 제어 모드
        self.control_modes = {
            'VR': 'vr_control',
            'MANUAL': 'manual_control', 
            'COLLECTION': 'data_collection'
        }
        self.current_mode = self.control_modes['MANUAL']
        
        # 로봇 상태
        self.manual_joints = [0.0, -0.3, 0.8, 0.0]
        self.manual_gripper = -0.01
        self.vr_joints = [0.0, -0.3, 0.8, 0.0]
        self.vr_gripper = -0.01
        self.selected_joint = 0
        
        # 데이터 수집
        self.data_collection_mode = False
        self.collection_count = 0
        
        # 성능 모니터링
        self.last_status_time = time.time()
        self.vr_status = {
            'calibrated': False,
            'trigger_value': 0.0,
            'total_points': 0,
            'session_collected': 0
        }
        
        # 액추에이터 매핑
        self.joint_mapping = {}
        self.gripper_mapping = -1
        self.setup_actuator_mapping()
        
        # GLFW 설정
        self.setup_glfw()
        
        # VR 브릿지 연결 시도
        self.connect_to_bridge()
        
        print("✅ GLFW 데이터 수집 컨트롤러 준비 완료!")
    
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
                    self.joint_mapping[joint_idx] = -1
            except:
                self.joint_mapping[joint_idx] = -1
        
        # 그리퍼 매핑
        try:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_gripper_joint')
            if actuator_id >= 0:
                self.gripper_mapping = actuator_id
                print(f"✅ 그리퍼 → 액추에이터 {actuator_id}")
            else:
                self.gripper_mapping = -1
        except:
            self.gripper_mapping = -1
    
    def setup_glfw(self):
        """GLFW 윈도우 및 콘텍스트 설정"""
        print("🖼️ GLFW 설정 중...")
        
        # GLFW 초기화
        glfw.init()
        self.window = glfw.create_window(1200, 900, "🎯 VR 데이터 수집 컨트롤러", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # MuJoCo 시각화 콘텍스트
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # 카메라 설정
        self.cam.distance = 1.5
        self.cam.elevation = -25.0
        self.cam.azimuth = 45.0
        self.cam.lookat = np.array([0.2, 0.0, 0.2])
        
        # 마우스 상태
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        
        # 콜백 등록
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        print("✅ GLFW 설정 완료")
    
    def connect_to_bridge(self):
        """VR 브릿지에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', 12345))
            self.socket.settimeout(0.001)
            self.connected = True
            print("✅ VR 브릿지 연결 성공!")
            
            # VR 데이터 수신 스레드 시작
            self.vr_thread = threading.Thread(target=self.vr_data_loop, daemon=True)
            self.vr_thread.start()
            
        except Exception as e:
            print(f"⚠️ VR 브릿지 연결 실패: {e}")
            print("🎮 수동 제어 모드로 시작합니다")
            self.connected = False
    
    def vr_data_loop(self):
        """VR 데이터 수신 루프"""
        while self.connected:
            try:
                raw_data = self.socket.recv(8192).decode('utf-8', errors='ignore')
                if not raw_data:
                    continue
                
                self.data_buffer += raw_data
                
                while '\n' in self.data_buffer:
                    line, self.data_buffer = self.data_buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            data = json.loads(line)
                            
                            # VR 상태 업데이트
                            if 'vr_status' in data:
                                self.vr_status.update(data['vr_status'])
                            
                            # 수집 상태 업데이트
                            if 'collection_status' in data:
                                cs = data['collection_status']
                                self.data_collection_mode = cs.get('mode_active', False)
                                self.vr_status['total_points'] = cs.get('total_points', 0)
                                self.vr_status['session_collected'] = cs.get('session_collected', 0)
                            
                            # 조인트 데이터 업데이트
                            if 'joint_angles' in data:
                                self.vr_joints = data['joint_angles'][:4]
                                self.vr_gripper = data.get('gripper', -0.01)
                                
                        except json.JSONDecodeError:
                            continue
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ VR 데이터 수신 오류: {e}")
                self.connected = False
                break
            
            time.sleep(0.001)
    
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
    
    def keyboard_callback(self, window, key, scancode, act, mods):
        """키보드 입력 처리 (GLFW 방식)"""
        if act == glfw.PRESS:
            
            # ESC - 종료
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            
            # 모드 전환
            elif key == glfw.KEY_V:  # V - VR 모드
                if self.connected:
                    self.current_mode = self.control_modes['VR']
                    print("🎮 VR 제어 모드")
                else:
                    print("❌ VR 브릿지가 연결되지 않음")
            
            elif key == glfw.KEY_M:  # M - 수동 모드
                self.current_mode = self.control_modes['MANUAL']
                print("⌨️ 수동 제어 모드")
            
            elif key == glfw.KEY_C:  # C - 데이터 수집 모드
                self.current_mode = self.control_modes['COLLECTION']
                self.data_collection_mode = not self.data_collection_mode
                
                # VR 브릿지에 알림
                self.send_to_bridge({
                    'command': 'data_collection_mode',
                    'enabled': self.data_collection_mode
                })
                
                mode_str = "활성화" if self.data_collection_mode else "비활성화"
                print(f"🎯 데이터 수집 모드 {mode_str}")
            
            # 조인트 선택 (숫자 키)
            elif key == glfw.KEY_1:
                self.selected_joint = 0
                print("🎯 Joint 1 선택됨")
            elif key == glfw.KEY_2:
                self.selected_joint = 1
                print("🎯 Joint 2 선택됨")
            elif key == glfw.KEY_3:
                self.selected_joint = 2
                print("🎯 Joint 3 선택됨")
            elif key == glfw.KEY_4:
                self.selected_joint = 3
                print("🎯 Joint 4 선택됨")
            
            # 조인트 제어 (화살표 키)
            elif key == glfw.KEY_UP:
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.adjust_joint(self.selected_joint, +0.1)
            
            elif key == glfw.KEY_DOWN:
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.adjust_joint(self.selected_joint, -0.1)
            
            # 그리퍼 제어
            elif key == glfw.KEY_O:  # O - 그리퍼 열기
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.manual_gripper = 0.019
                    print("🤏 그리퍼 열기")
            
            elif key == glfw.KEY_P:  # P - 그리퍼 닫기
                if self.current_mode in [self.control_modes['MANUAL'], self.control_modes['COLLECTION']]:
                    self.manual_gripper = -0.01
                    print("✊ 그리퍼 닫기")
            
            # 데이터 수집
            elif key == glfw.KEY_SPACE:  # SPACE - 매핑 저장
                if self.current_mode == self.control_modes['COLLECTION'] and self.data_collection_mode:
                    self.collect_current_mapping()
            
            # 리셋
            elif key == glfw.KEY_R:  # R - 리셋
                self.manual_joints = [0.0, -0.3, 0.8, 0.0]
                self.manual_gripper = -0.01
                print("🔄 수동 조인트 리셋")
            
            # 저장
            elif key == glfw.KEY_S:  # S - 저장
                self.send_to_bridge({'command': 'save_data'})
                print("💾 매핑 데이터 저장 요청")
    
    def adjust_joint(self, joint_idx, delta):
        """조인트 값 조정"""
        if 0 <= joint_idx < 4:
            self.manual_joints[joint_idx] += delta
            
            # 조인트 제한 적용
            joint_limits = [
                [-3.14, 3.14],  # joint1
                [-1.5, 1.5],    # joint2
                [-1.5, 1.4],    # joint3
                [-1.7, 1.97]    # joint4
            ]
            
            limits = joint_limits[joint_idx]
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
        """현재 매핑 데이터 수집"""
        self.send_to_bridge({'command': 'collect_data'})
        self.collection_count += 1
        print(f"📊 매핑 데이터 수집 요청 #{self.collection_count}")
    
    def mouse_button_callback(self, window, button, act, mods):
        """마우스 버튼 콜백"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.button_left = (act == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.button_middle = (act == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.button_right = (act == glfw.PRESS)
        
        self.lastx, self.lasty = glfw.get_cursor_pos(window)
    
    def mouse_move_callback(self, window, xpos, ypos):
        """마우스 이동 콜백"""
        if self.button_left:
            # 카메라 회전
            dy = 0.01 * (ypos - self.lasty)
            dx = 0.01 * (xpos - self.lastx)
            self.cam.elevation = np.clip(self.cam.elevation - dy*100, -90, 90)
            self.cam.azimuth = (self.cam.azimuth + dx*100) % 360
        elif self.button_middle:
            # 카메라 팬
            dx = 0.001 * (xpos - self.lastx)
            dy = 0.001 * (ypos - self.lasty)
            self.cam.lookat[0] += -dx * self.cam.distance
            self.cam.lookat[1] += dy * self.cam.distance
        elif self.button_right:
            # 카메라 줌
            dy = 0.01 * (ypos - self.lasty)
            self.cam.distance = np.clip(self.cam.distance + dy, 0.1, 5.0)
        
        self.lastx = xpos
        self.lasty = ypos
    
    def scroll_callback(self, window, xoffset, yoffset):
        """스크롤 콜백"""
        self.cam.distance = np.clip(self.cam.distance - 0.1 * yoffset, 0.1, 5.0)
    
    def update_robot(self):
        """로봇 상태 업데이트"""
        if self.current_mode == self.control_modes['VR']:
            target_joints = self.vr_joints
            target_gripper = self.vr_gripper
        else:
            target_joints = self.manual_joints
            target_gripper = self.manual_gripper
        
        # 조인트 업데이트
        for joint_idx, angle in enumerate(target_joints[:4]):
            actuator_id = self.joint_mapping.get(joint_idx, -1)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = angle
        
        # 그리퍼 업데이트
        if self.gripper_mapping >= 0:
            self.data.ctrl[self.gripper_mapping] = target_gripper
    
    def render_scene(self):
        """화면 렌더링"""
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # 씬 업데이트
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        
        # 렌더링
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # UI 오버레이
        mode_names = {
            'vr_control': 'VR 제어',
            'manual_control': '수동 제어',
            'data_collection': '데이터 수집'
        }
        mode_name = mode_names.get(self.current_mode, '알 수 없음')
        
        status_text = [
            "🎯 === VR 데이터 수집 컨트롤러 ===",
            f"🎮 제어 모드: {mode_name}",
            f"🔗 VR 연결: {'✅' if self.connected else '❌'}",
            f"🎯 수집 모드: {'✅ 활성' if self.data_collection_mode else '❌ 비활성'}",
            f"📈 총 매핑: {self.vr_status.get('total_points', 0)}개",
            f"📍 세션 수집: {self.vr_status.get('session_collected', 0)}개",
            "",
            "⌨️ 키보드 조작:",
            "  V: VR 모드 | M: 수동 모드 | C: 수집 모드",
            "  1-4: 조인트 선택 | ↑↓: 조인트 제어",
            "  O: 그리퍼 열기 | P: 그리퍼 닫기",
            "  SPACE: 매핑 저장 | R: 리셋 | S: 저장",
            "  ESC: 종료",
            "",
            f"🤖 수동 조인트: {[f'{j:.2f}' for j in self.manual_joints]}",
            f"🎯 선택된 조인트: Joint{self.selected_joint + 1}",
            f"🤏 그리퍼: {self.manual_gripper:.3f}",
        ]
        
        overlay = "\n".join(status_text)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            overlay,
            "",
            self.context
        )
        
        # 버퍼 스왑
        glfw.swap_buffers(self.window)
    
    def print_status(self):
        """주기적 상태 출력"""
        current_time = time.time()
        if current_time - self.last_status_time > 5.0:  # 5초마다
            
            mode_names = {
                'vr_control': 'VR 제어',
                'manual_control': '수동 제어',
                'data_collection': '데이터 수집'
            }
            mode_name = mode_names.get(self.current_mode, '알 수 없음')
            
            print(f"\n🎯 === 상태 요약 ===")
            print(f"🎮 모드: {mode_name}")
            print(f"🔗 VR: {'연결됨' if self.connected else '연결 안됨'}")
            print(f"🎯 수집: {'활성' if self.data_collection_mode else '비활성'}")
            print(f"📊 매핑: 총 {self.vr_status.get('total_points', 0)}개, 세션 {self.vr_status.get('session_collected', 0)}개")
            print(f"🤖 조인트: {[f'{j:.2f}' for j in self.manual_joints]}")
            print(f"🎯 선택: Joint{self.selected_joint + 1}")
            print("-" * 50)
            
            self.last_status_time = current_time
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎯 === GLFW VR 데이터 수집 컨트롤러 시작! ===")
        print("=" * 60)
        print("🎮 제어 방법:")
        print("  V: VR 제어 모드")
        print("  M: 수동 제어 모드")
        print("  C: 데이터 수집 모드 토글")
        print("  1-4: 조인트 선택")
        print("  ↑/↓: 선택된 조인트 제어")
        print("  O: 그리퍼 열기")
        print("  P: 그리퍼 닫기")
        print("  SPACE: 현재 매핑 저장")
        print("  R: 수동 조인트 리셋")
        print("  S: 매핑 데이터 저장")
        print("  ESC: 종료")
        print("=" * 60)
        print("🎯 데이터 수집 방법:")
        print("  1. C키로 수집 모드 활성화")
        print("  2. 키보드로 로봇을 원하는 자세로 조정")
        print("  3. VR 컨트롤러를 해당 위치로 이동")
        print("  4. SPACE키로 매핑 저장")
        print("=" * 60)
        
        try:
            frame_count = 0
            
            while not glfw.window_should_close(self.window):
                frame_start = time.time()
                
                # 로봇 업데이트
                self.update_robot()
                
                # 물리 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                
                # 화면 렌더링
                self.render_scene()
                
                # 이벤트 처리
                glfw.poll_events()
                
                # 주기적 상태 출력
                if frame_count % 300 == 0:  # 5초마다
                    self.print_status()
                
                frame_count += 1
                
                # 프레임 제한
                frame_time = time.time() - frame_start
                time.sleep(max(0, 0.016 - frame_time))  # ~60 FPS
                
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 중단됨")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("🧹 리소스 정리 중...")
        
        if self.connected and self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        glfw.terminate()
        print("✅ 정리 완료")

if __name__ == "__main__":
    try:
        print("🎯 GLFW 데이터 수집 시스템 시작...")
        collector = GLFWDataCollector()
        collector.run()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 시스템 종료")
