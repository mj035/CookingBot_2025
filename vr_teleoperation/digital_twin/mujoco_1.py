#!/usr/bin/env python3
"""
MuJoCo 양팔 시각화 - 실물 양팔 움직임을 실시간으로 표시
dual_arm_teaching.py와 함께 사용하여 동기화 확인
"""

import socket
import json
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque

class DualArmVisualizer:
    def __init__(self):
        print("🎮 MuJoCo Dual Arm Visualizer")
        print("📡 실물 양팔 데이터 대기 중... (포트 12345)")
        print("📊 양팔 동기화 테스트용\n")

        # MuJoCo 모델 로드 (scene_dual.xml 사용)
        self.model = mujoco.MjModel.from_xml_path('../dual_arm/scene_dual.xml')
        self.data = mujoco.MjData(self.model)

        # 왼팔 액추에이터 매핑
        self.left_actuator_ids = []
        print("=== 왼팔 액추에이터 매핑 ===")
        left_actuator_names = ['actuator_joint1', 'actuator_joint2',
                               'actuator_joint3', 'actuator_joint4']
        for name in left_actuator_names:
            try:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.left_actuator_ids.append(act_id)
                print(f"  ✅ {name} → ID: {act_id}")
            except:
                print(f"  ❌ {name} not found")
                self.left_actuator_ids.append(-1)

        # 왼팔 그리퍼
        try:
            self.left_gripper_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_gripper_joint')
            print(f"  ✅ actuator_gripper_joint → ID: {self.left_gripper_id}")
        except:
            self.left_gripper_id = -1

        # 오른팔 액추에이터 매핑
        self.right_actuator_ids = []
        print("\n=== 오른팔 액추에이터 매핑 ===")
        right_actuator_names = ['actuator_joint1_r', 'actuator_joint2_r',
                                'actuator_joint3_r', 'actuator_joint4_r']
        for name in right_actuator_names:
            try:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.right_actuator_ids.append(act_id)
                print(f"  ✅ {name} → ID: {act_id}")
            except:
                print(f"  ❌ {name} not found")
                self.right_actuator_ids.append(-1)

        # 오른팔 그리퍼
        try:
            self.right_gripper_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator_gripper_joint_r')
            print(f"  ✅ actuator_gripper_joint_r → ID: {self.right_gripper_id}")
        except:
            self.right_gripper_id = -1

        # 로봇 상태 (MuJoCo 초기 자세)
        self.left_joints = [0.0, 0.0, 0.0, 0.0]
        self.left_gripper = 0.019
        self.right_joints = [0.0, 0.0, 0.0, 0.0]
        self.right_gripper = 0.019
        self.data_received = False

        # 초기 자세 적용
        self.set_initial_pose()

        # 소켓 서버
        self.setup_socket_server()

        # 통계
        self.frame_count = 0
        self.last_print_time = time.time()

        print("\n✅ 시각화 준비 완료\n")

    def set_initial_pose(self):
        """초기 자세 설정"""
        print("🤖 초기 자세 설정 중...")

        # 왼팔 초기 자세
        for i, act_id in enumerate(self.left_actuator_ids):
            if act_id >= 0:
                self.data.ctrl[act_id] = self.left_joints[i]
        if self.left_gripper_id >= 0:
            self.data.ctrl[self.left_gripper_id] = self.left_gripper

        # 오른팔 초기 자세
        for i, act_id in enumerate(self.right_actuator_ids):
            if act_id >= 0:
                self.data.ctrl[act_id] = self.right_joints[i]
        if self.right_gripper_id >= 0:
            self.data.ctrl[self.right_gripper_id] = self.right_gripper

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

                                        # 왼팔 데이터 처리
                                        if 'left_arm' in msg:
                                            if 'joint_angles' in msg['left_arm']:
                                                raw_joints = msg['left_arm']['joint_angles'][:4]
                                                # 안전 범위 제한
                                                safe_limits = [
                                                    (-1.57, 1.57),  # Joint1
                                                    (-1.5, 1.5),    # Joint2
                                                    (-1.5, 1.4),    # Joint3
                                                    (-1.7, 1.97)    # Joint4
                                                ]
                                                self.left_joints = [
                                                    max(safe_limits[i][0], min(safe_limits[i][1], raw_joints[i]))
                                                    for i in range(4)
                                                ]
                                            if 'gripper' in msg['left_arm']:
                                                self.left_gripper = msg['left_arm']['gripper']

                                        # 오른팔 데이터 처리
                                        if 'right_arm' in msg:
                                            if 'joint_angles' in msg['right_arm']:
                                                raw_joints = msg['right_arm']['joint_angles'][:4]
                                                # 안전 범위 제한
                                                safe_limits = [
                                                    (-1.57, 1.57),  # Joint1
                                                    (-1.5, 1.5),    # Joint2
                                                    (-1.5, 1.4),    # Joint3
                                                    (-1.7, 1.97)    # Joint4
                                                ]
                                                self.right_joints = [
                                                    max(safe_limits[i][0], min(safe_limits[i][1], raw_joints[i]))
                                                    for i in range(4)
                                                ]
                                            if 'gripper' in msg['right_arm']:
                                                self.right_gripper = msg['right_arm']['gripper']

                                        self.data_received = True

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
            # 카메라 설정 (양팔이 모두 보이도록)
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

            while viewer.is_running():
                # 왼팔 액추에이터 제어
                for i, act_id in enumerate(self.left_actuator_ids):
                    if act_id >= 0 and i < len(self.left_joints):
                        self.data.ctrl[act_id] = self.left_joints[i]

                # 왼팔 그리퍼
                if self.left_gripper_id >= 0:
                    self.data.ctrl[self.left_gripper_id] = self.left_gripper

                # 오른팔 액추에이터 제어
                for i, act_id in enumerate(self.right_actuator_ids):
                    if act_id >= 0 and i < len(self.right_joints):
                        self.data.ctrl[act_id] = self.right_joints[i]

                # 오른팔 그리퍼
                if self.right_gripper_id >= 0:
                    self.data.ctrl[self.right_gripper_id] = self.right_gripper

                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 상태 출력 (1초마다)
                current_time = time.time()
                if current_time - self.last_print_time > 1.0:
                    if self.data_received:
                        left_str = ', '.join([f'{j:.2f}' for j in self.left_joints])
                        right_str = ', '.join([f'{j:.2f}' for j in self.right_joints])
                        print(f"📊 왼팔: [{left_str}] | 오른팔: [{right_str}]")
                    else:
                        print("⏳ 실물 양팔 데이터 대기 중...")
                    self.last_print_time = current_time

                self.frame_count += 1
                time.sleep(0.002)  # ~500 FPS

            print("🏁 시각화 종료")

def main():
    try:
        visualizer = DualArmVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n중단됨")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()