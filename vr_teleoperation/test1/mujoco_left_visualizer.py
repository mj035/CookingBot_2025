#!/usr/bin/env python3
"""
MuJoCo 왼팔 시각화 - 실물 로봇 움직임을 실시간으로 표시
"""

import socket
import json
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque

class LeftArmVisualizer:
    def __init__(self):
        print("🎮 MuJoCo Left Arm Visualizer")
        print("📡 실물 로봇 데이터 대기 중... (포트 12345)")
        
        # MuJoCo 모델 로드 (왼팔만 사용)
        self.model = mujoco.MjModel.from_xml_path('../single_arm/omx.xml')
        self.data = mujoco.MjData(self.model)
        
        # 조인트 인덱스 찾기
        self.joint_ids = []
        for i in range(1, 5):
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
                self.joint_ids.append(joint_id)
                print(f"  Joint{i} ID: {joint_id}")
            except:
                print(f"  Joint{i} not found")
        
        # 그리퍼 조인트
        try:
            self.gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_joint')
        except:
            self.gripper_id = -1
        
        # 로봇 상태
        self.robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.robot_gripper = -0.01
        self.data_received = False
        
        # 소켓 서버
        self.setup_socket_server()
        
        # 통계
        self.frame_count = 0
        self.last_print_time = time.time()
        
        print("✅ 시각화 준비 완료\n")
    
    def setup_socket_server(self):
        """소켓 서버 설정"""
        def server_thread():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', 12345))
            server.listen(1)
            
            while True:
                try:
                    client, addr = server.accept()
                    print(f"🔗 Teaching 클라이언트 연결: {addr}")
                    
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
                                                self.robot_joints = msg['left_arm']['joint_angles'][:4]
                                                self.data_received = True
                                            if 'gripper' in msg['left_arm']:
                                                self.robot_gripper = msg['left_arm']['gripper']
                                    except json.JSONDecodeError:
                                        pass
                        except socket.timeout:
                            continue
                        except Exception as e:
                            break
                    
                    print("⚠️ Teaching 클라이언트 연결 끊김")
                    client.close()
                    
                except Exception as e:
                    print(f"서버 오류: {e}")
                    time.sleep(1)
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def run(self):
        """메인 루프"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 카메라 설정
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            
            while viewer.is_running():
                # 로봇 조인트 값 적용
                for i, joint_id in enumerate(self.joint_ids):
                    if joint_id >= 0 and i < len(self.robot_joints):
                        self.data.qpos[joint_id] = self.robot_joints[i]
                
                # 그리퍼 적용
                if self.gripper_id >= 0:
                    self.data.qpos[self.gripper_id] = self.robot_gripper
                
                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 상태 출력 (1초마다)
                current_time = time.time()
                if current_time - self.last_print_time > 1.0:
                    if self.data_received:
                        joints_str = ', '.join([f'{j:.2f}' for j in self.robot_joints])
                        print(f"📊 조인트: [{joints_str}] | 그리퍼: {self.robot_gripper:.2f}")
                    else:
                        print("⏳ 실물 로봇 데이터 대기 중...")
                    self.last_print_time = current_time
                
                self.frame_count += 1
                time.sleep(0.002)  # ~500 FPS
            
            print("🏁 시각화 종료")

def main():
    try:
        visualizer = LeftArmVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n중단됨")
    except Exception as e:
        print(f"오류: {e}")

if __name__ == '__main__':
    main()