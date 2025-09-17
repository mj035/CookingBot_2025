#!/usr/bin/env python3
"""
MuJoCo 오른팔 시각화 - Single Arm Scene 사용 (왼팔과 동일한 방식)
실물 로봇 움직임을 실시간으로 표시
"""

import socket
import json
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque

class RightArmSingleVisualizer:
    def __init__(self):
        print("🎮 MuJoCo Right Arm Single Visualizer")
        print("📡 실물 오른팔 데이터 대기 중... (포트 12345)")
        print("📊 Single arm scene 사용 (왼팔과 동일한 방식)\n")

        # MuJoCo 모델 로드 - single arm scene 사용!
        self.model = mujoco.MjModel.from_xml_path('../single_arm/scene.xml')
        self.data = mujoco.MjData(self.model)

        # 조인트 인덱스 찾기 (왼팔과 동일)
        self.joint_ids = []
        print("\n=== 조인트 매핑 ===")
        for i in range(1, 5):
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
                self.joint_ids.append(joint_id)
                print(f"  ✅ Joint{i} → ID: {joint_id}")
            except:
                print(f"  ❌ Joint{i} not found")
                self.joint_ids.append(-1)

        # 그리퍼 조인트
        try:
            self.gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_joint')
            print(f"  ✅ gripper_joint → ID: {self.gripper_id}")
        except:
            self.gripper_id = -1

        # 액추에이터 매핑 (왼팔과 동일한 이름 사용)
        self.actuator_ids = []
        print("\n=== 액추에이터 매핑 ===")
        actuator_names = ['actuator_joint1', 'actuator_joint2', 'actuator_joint3', 'actuator_joint4']
        for name in actuator_names:
            try:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.actuator_ids.append(act_id)
                print(f"  ✅ {name} → ID: {act_id}")
            except:
                print(f"  ❌ {name} not found")
                self.actuator_ids.append(-1)

        # 그리퍼 액추에이터
        try:
            self.gripper_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_gripper_joint')
            print(f"  ✅ actuator_gripper_joint → ID: {self.gripper_act_id}")
        except:
            self.gripper_act_id = -1

        # 로봇 상태 (MuJoCo 초기 자세 - 모두 0)
        self.robot_joints = [0.0, 0.0, 0.0, 0.0]  # 초기 자세
        self.robot_gripper = 0.019  # 그리퍼 열림
        self.data_received = False

        # 초기 자세 적용
        self.set_initial_pose()

        # 소켓 서버
        self.setup_socket_server()

        # 통계
        self.frame_count = 0
        self.last_print_time = time.time()

        print("✅ 시각화 준비 완료\n")

    def set_initial_pose(self):
        """초기 자세 설정"""
        print("🤖 초기 자세 설정 중...")
        for i, act_id in enumerate(self.actuator_ids):
            if act_id >= 0:
                self.data.ctrl[act_id] = self.robot_joints[i]

        # 시뮬레이션 스텝 실행하여 초기 자세 적용
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        print("✅ 초기 자세 설정 완료")

    def setup_socket_server(self):
        """소켓 서버 설정"""
        def server_thread():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                # 포트가 사용 중이면 다른 포트 시도
                try:
                    server.bind(('localhost', 12345))
                    print("📡 포트 12345에서 대기 중")
                except OSError as e:
                    print(f"⚠️ 포트 12345 사용 중: {e}")
                    print("💡 포트 12346 시도...")
                    server.bind(('localhost', 12346))
                    print("✅ 포트 12346 사용")

                server.listen(1)
                print("📡 소켓 서버 시작됨")
            except Exception as e:
                print(f"❌ 서버 시작 실패: {e}")
                return

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

                                        # 오른팔 데이터 처리
                                        if 'right_arm' in msg:
                                            if 'joint_angles' in msg['right_arm']:
                                                raw_joints = msg['right_arm']['joint_angles'][:4]
                                                # 안전 범위 제한
                                                safe_limits = [
                                                    (-1.57, 1.57),  # Joint1: ±90도
                                                    (-1.5, 1.5),    # Joint2
                                                    (-1.5, 1.4),    # Joint3
                                                    (-1.7, 1.97)    # Joint4
                                                ]
                                                self.robot_joints = [
                                                    max(safe_limits[i][0], min(safe_limits[i][1], raw_joints[i]))
                                                    for i in range(4)
                                                ]
                                                self.data_received = True
                                            if 'gripper' in msg['right_arm']:
                                                self.robot_gripper = msg['right_arm']['gripper']
                                    except json.JSONDecodeError as e:
                                        print(f"❌ JSON 파싱 오류: {e}")
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
                # 액추에이터를 통한 제어
                for i, act_id in enumerate(self.actuator_ids):
                    if act_id >= 0 and i < len(self.robot_joints):
                        self.data.ctrl[act_id] = self.robot_joints[i]

                # 그리퍼 액추에이터
                if self.gripper_act_id >= 0:
                    self.data.ctrl[self.gripper_act_id] = self.robot_gripper

                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 상태 출력 (1초마다)
                current_time = time.time()
                if current_time - self.last_print_time > 1.0:
                    if self.data_received:
                        joints_str = ', '.join([f'{j:.2f}' for j in self.robot_joints])
                        print(f"📊 오른팔 조인트: [{joints_str}] | 그리퍼: {self.robot_gripper:.2f}")
                    else:
                        print("⏳ 실물 오른팔 데이터 대기 중...")
                    self.last_print_time = current_time

                self.frame_count += 1
                time.sleep(0.002)  # ~500 FPS

            print("🏁 시각화 종료")

def main():
    try:
        visualizer = RightArmSingleVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n중단됨")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()