#!/usr/bin/env python3
"""
🎯 GLFW 기반 데이터 수집 MuJoCo 컨트롤러 - 키 코드 수정됨
"""

import numpy as np
import mujoco
import time
import socket
import json
import threading
from mujoco.glfw import glfw

class FixedMuJoCoController:
    def __init__(self):
        print("🎯 Fixed MuJoCo Controller 초기화 중...")
        
        # 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
        # 제어 상태
        self.manual_joints = [0.0, -0.3, 0.8, 0.0]
        self.manual_gripper = -0.01
        self.vr_joints = [0.0, -0.3, 0.8, 0.0]
        self.vr_gripper = -0.01
        self.selected_joint = 0
        
        # 모드 및 데이터 수집
        self.control_mode = 'MANUAL'  # VR, MANUAL, COLLECTION
        self.data_collection_mode = False
        self.collection_count = 0
        
        # VR 연결
        self.socket = None
        self.connected = False
        self.data_buffer = ""
        self.vr_status = {'calibrated': False, 'total_points': 0, 'session_collected': 0}
        
        # 액추에이터 매핑
        self.joint_mapping = {}
        self.gripper_mapping = -1
        self.setup_actuator_mapping()
        
        # GLFW 설정
        self.setup_glfw()
        
        # VR 브릿지 연결 시도
        self.connect_to_bridge()
        
        print("✅ Fixed MuJoCo Controller 준비 완료!")
    
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
                    self.joint_mapping[i] = -1
            except:
                self.joint_mapping[i] = -1
        
        # 그리퍼
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
        """GLFW 윈도우 설정"""
        glfw.init()
        self.window = glfw.create_window(1200, 900, "🎯 VR 데이터 수집 컨트롤러", None, None)
        glfw.make_context_current(self.window)
        
        # 렌더링 콘텍스트
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # 카메라 설정
        self.cam.distance = 1.5
        self.cam.elevation = -25
        self.cam.azimuth = 45
        self.cam.lookat = np.array([0.2, 0, 0.2])
        
        # 마우스 상태
        self.button_left = False
        self.lastx = 0
        self.lasty = 0
        
        # 콜백 등록
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
    
    def connect_to_bridge(self):
        """VR 브릿지 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', 12345))
            self.socket.settimeout(0.001)
            self.connected = True
            print("✅ VR 브릿지 연결 성공!")
            
            # VR 데이터 수신 스레드
            threading.Thread(target=self.vr_data_loop, daemon=True).start()
            
        except Exception as e:
            print(f"⚠️ VR 브릿지 연결 실패: {e} - 수동 모드로 시작")
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
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            
                            # VR 상태 업데이트
                            if 'vr_status' in data:
                                self.vr_status.update(data['vr_status'])
                            
                            # 수집 상태 업데이트
                            if 'collection_status' in data:
                                cs = data['collection_status']
                                self.data_collection_mode = cs.get('mode_active', False)
                                self.vr_status['total_points'] = cs.get('total_points', 0)
                                self.vr_status['session_collected'] = cs.get('session_collected', 0)
                            
                            # 조인트 데이터
                            if 'joint_angles' in data:
                                self.vr_joints = data['joint_angles'][:4]
                                self.vr_gripper = data.get('gripper', -0.01)
                                
                        except json.JSONDecodeError:
                            continue
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ VR 데이터 오류: {e}")
                self.connected = False
                break
            
            time.sleep(0.001)
    
    def send_to_bridge(self, command_data):
        """VR 브릿지로 명령 전송"""
        if self.connected:
            try:
                message = json.dumps(command_data) + '\n'
                self.socket.sendall(message.encode())
            except Exception as e:
                print(f"⚠️ 브릿지 전송 오류: {e}")
                self.connected = False
    
    def keyboard_callback(self, window, key, scancode, act, mods):
        """키보드 콜백 - 확인된 키 코드 사용"""
        if act == glfw.PRESS:
            
            # ESC - 종료
            if key == 256:  # ESC
                glfw.set_window_should_close(window, True)
            
            # 모드 전환
            elif key == 86:  # V
                if self.connected:
                    self.control_mode = 'VR'
                    print("🎮 VR 제어 모드")
                else:
                    print("❌ VR 브릿지 연결 안됨")
            
            elif key == 77:  # M
                self.control_mode = 'MANUAL'
                print("⌨️ 수동 제어 모드")
            
            elif key == 67:  # C (확인된 키 코드)
                self.control_mode = 'COLLECTION'
                self.data_collection_mode = not self.data_collection_mode
                
                self.send_to_bridge({
                    'command': 'data_collection_mode',
                    'enabled': self.data_collection_mode
                })
                
                mode_str = "활성화" if self.data_collection_mode else "비활성화"
                print(f"🎯 데이터 수집 모드 {mode_str}")
            
            # 조인트 선택 (숫자 키)
            elif key == 49:  # 1
                self.selected_joint = 0
                print("🎯 Joint 1 선택됨")
            elif key == 50:  # 2
                self.selected_joint = 1
                print("🎯 Joint 2 선택됨")
            elif key == 51:  # 3
                self.selected_joint = 2
                print("🎯 Joint 3 선택됨")
            elif key == 52:  # 4
                self.selected_joint = 3
                print("🎯 Joint 4 선택됨")
            
            # 조인트 제어 (확인된 화살표 키 코드)
            elif key == 265:  # UP 화살표 (확인됨)
                if self.control_mode in ['MANUAL', 'COLLECTION']:
                    self.adjust_joint(self.selected_joint, +0.1)
            
            elif key == 264:  # DOWN 화살표 (확인됨)
                if self.control_mode in ['MANUAL', 'COLLECTION']:
                    self.adjust_joint(self.selected_joint, -0.1)
            
            # 그리퍼 제어
            elif key == 79:  # O
                if self.control_mode in ['MANUAL', 'COLLECTION']:
                    self.manual_gripper = 0.019
                    print("🤏 그리퍼 열기")
            
            elif key == 80:  # P
                if self.control_mode in ['MANUAL', 'COLLECTION']:
                    self.manual_gripper = -0.01
                    print("✊ 그리퍼 닫기")
            
            # 데이터 수집
            elif key == 32:  # SPACE
                if self.control_mode == 'COLLECTION' and self.data_collection_mode:
                    self.collect_current_mapping()
            
            # 리셋
            elif key == 82:  # R
                self.manual_joints = [0.0, -0.3, 0.8, 0.0]
                self.manual_gripper = -0.01
                print("🔄 수동 조인트 리셋")
            
            # 저장
            elif key == 83:  # S
                self.send_to_bridge({'command': 'save_data'})
                print("💾 매핑 데이터 저장 요청")
            
            # 디버그용 - 키 코드 출력
            else:
                char = chr(key) if 32 <= key <= 126 else 'special'
                print(f"🔍 키 눌림: key={key}, 문자={char}")
    
    def adjust_joint(self, joint_idx, delta):
        """조인트 값 조정"""
        if 0 <= joint_idx < 4:
            self.manual_joints[joint_idx] += delta
            
            # 조인트 제한
            limits = [[-3.14, 3.14], [-1.5, 1.5], [-1.5, 1.4], [-1.7, 1.97]]
            joint_limits = limits[joint_idx]
            self.manual_joints[joint_idx] = np.clip(
                self.manual_joints[joint_idx], joint_limits[0], joint_limits[1]
            )
            
            print(f"🔧 Joint{joint_idx+1}: {self.manual_joints[joint_idx]:.2f}")
            
            # VR 브릿지에 전송
            self.send_to_bridge({
                'command': 'set_robot_joints',
                'joints': self.manual_joints
            })
    
    def collect_current_mapping(self):
        """현재 매핑 데이터 수집"""
        self.send_to_bridge({'command': 'collect_data'})
        self.collection_count += 1
        print(f"📊 매핑 데이터 수집 #{self.collection_count}")
    
    def mouse_button_callback(self, window, button, act, mods):
        """마우스 버튼 콜백"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.button_left = (act == glfw.PRESS)
        self.lastx, self.lasty = glfw.get_cursor_pos(window)
    
    def mouse_move_callback(self, window, xpos, ypos):
        """마우스 이동 콜백"""
        if self.button_left:
            dy = 0.01 * (ypos - self.lasty)
            dx = 0.01 * (xpos - self.lastx)
            self.cam.elevation = np.clip(self.cam.elevation - dy*100, -90, 90)
            self.cam.azimuth = (self.cam.azimuth + dx*100) % 360
        
        self.lastx = xpos
        self.lasty = ypos
    
    def scroll_callback(self, window, xoffset, yoffset):
        """스크롤 콜백"""
        self.cam.distance = np.clip(self.cam.distance - 0.1 * yoffset, 0.1, 5.0)
    
    def update_robot(self):
        """로봇 상태 업데이트"""
        if self.control_mode == 'VR':
            target_joints = self.vr_joints
            target_gripper = self.vr_gripper
        else:
            target_joints = self.manual_joints
            target_gripper = self.manual_gripper
        
        # 조인트 업데이트
        for i, angle in enumerate(target_joints[:4]):
            actuator_id = self.joint_mapping.get(i, -1)
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
        
        # UI 표시
        ui_text = [
            "🎯 === VR 데이터 수집 컨트롤러 ===",
            f"🎮 모드: {self.control_mode}",
            f"🔗 VR: {'연결됨' if self.connected else '연결 안됨'}",
            f"🎯 수집: {'활성' if self.data_collection_mode else '비활성'}",
            f"📊 매핑: {self.vr_status.get('total_points', 0)}개",
            f"📍 세션: {self.vr_status.get('session_collected', 0)}개",
            "",
            "⌨️ 키보드 조작:",
            "V: VR모드 | M: 수동모드 | C: 수집모드",
            "1-4: 조인트선택 | ↑↓: 조인트제어",
            "O: 그리퍼열기 | P: 그리퍼닫기",
            "SPACE: 매핑저장 | R: 리셋 | S: 저장",
            "",
            f"🤖 조인트: {[f'{j:.2f}' for j in self.manual_joints]}",
            f"🎯 선택: Joint{self.selected_joint + 1}",
            f"🤏 그리퍼: {self.manual_gripper:.3f}",
        ]
        
        overlay = "\n".join(ui_text)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, overlay, "", self.context
        )
        
        glfw.swap_buffers(self.window)
    
    def run(self):
        """메인 실행 루프"""
        print("\n🎯 === Fixed MuJoCo Controller 시작! ===")
        print("🎮 V: VR모드 | M: 수동모드 | C: 수집모드")
        print("🔢 1-4: 조인트선택 | ↑↓: 조인트제어")
        print("🤏 O: 그리퍼열기 | P: 그리퍼닫기")
        print("📊 SPACE: 매핑저장 | R: 리셋 | S: 저장")
        print("=" * 50)
        
        try:
            while not glfw.window_should_close(self.window):
                # 로봇 업데이트
                self.update_robot()
                
                # 물리 시뮬레이션
                mujoco.mj_step(self.model, self.data)
                
                # 렌더링
                self.render_scene()
                
                # 이벤트 처리
                glfw.poll_events()
                
                # 프레임 제한
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("\n⚠️ 사용자 중단")
        finally:
            if self.connected and self.socket:
                self.socket.close()
            glfw.terminate()

if __name__ == "__main__":
    try:
        controller = FixedMuJoCoController()
        controller.run()
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
