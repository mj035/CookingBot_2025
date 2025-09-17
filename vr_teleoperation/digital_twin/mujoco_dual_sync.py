#!/usr/bin/env python3
"""
🎯 Digital Twin - 양팔 동기화 + VR 텔레오퍼레이션 + 실물 제어
- VR 입력 수신 (dual_arm_bridge_improved.py로부터)
- MuJoCo 시뮬레이션 (동기화된 상태)
- 실물 로봇 제어 (Dynamixel SDK)
"""

import socket
import json
import time
import threading
from collections import deque
import numpy as np
import mujoco
import mujoco.viewer
from dynamixel_sdk import *

# 경로/소켓 설정
XML_SCENE_PATH = '../dual_arm/scene_dual.xml'
BRIDGE_ADDR = ('localhost', 12345)

# Dynamixel 설정
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyACM0'

# Control table addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# 모터 ID
LEFT_ARM_IDS = [11, 12, 13, 14, 15]
RIGHT_ARM_IDS = [21, 22, 23, 24, 25]

# 카메라 설정
CAMERA_MODE = 'behind'
CAMERA_DISTANCE = 2.0
AZIMUTH_FRONT = 180
CAMERA_ELEVATION = -15

# 조인트 안전 범위
JOINT_LIMITS = {
    'j1': (-3.14, 3.14),
    'j2': (-1.5, 1.5),
    'j3': (-1.5, 1.4),
    'j4': (-1.7, 1.97)
}
GRIPPER_RANGE = (-0.01, 0.019)

class UnifiedBridgeClient:
    """VR 브릿지 클라이언트"""
    def __init__(self, addr):
        self.addr = addr
        self.sock = None
        self.connected = False
        self.buffer = ""
        self.latest_left = None
        self.latest_right = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(2.0)
                self.sock.connect(self.addr)
                self.sock.settimeout(0.001)
                self.connected = True
                print(f"🔗 VR Bridge 연결: {self.addr}")

                while True:
                    try:
                        raw = self.sock.recv(8192).decode('utf-8', errors='ignore')
                        if not raw:
                            raise ConnectionError("peer closed")
                        self.buffer += raw
                        while '\n' in self.buffer:
                            line, self.buffer = self.buffer.split('\n', 1)
                            s = line.strip()
                            if not s:
                                continue
                            try:
                                d = json.loads(s)
                                if 'left_arm' in d:
                                    self.latest_left = d['left_arm']
                                if 'right_arm' in d:
                                    self.latest_right = d['right_arm']
                            except json.JSONDecodeError:
                                continue
                    except socket.timeout:
                        pass
            except Exception as e:
                if self.connected:
                    print(f"⚠️ VR Bridge 연결 끊김: {e}")
                self.connected = False
                time.sleep(1.0)
            finally:
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                self.sock = None

    def pop_latest_left(self):
        d = self.latest_left
        self.latest_left = None
        return d

    def pop_latest_right(self):
        d = self.latest_right
        self.latest_right = None
        return d

class DualArmDigitalTwin:
    """양팔 디지털 트윈 컨트롤러"""
    def __init__(self):
        print("🤖 Digital Twin - Dual Arm Controller")
        print("📡 VR 입력 + MuJoCo 시뮬레이션 + 실물 제어")

        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(XML_SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        # 오프셋 (동기화용)
        self.left_offsets = [0.0, -0.43, 1.94, -0.42]
        self.right_offsets = [0.0, -0.43, 1.94, -0.42]  # 실측값으로 변경

        # 액추에이터 매핑
        self.left_map = self._map_actuators(side="L")
        self.right_map = self._map_actuators(side="R")

        # Dynamixel 초기화
        self.setup_dynamixel()

        # 초기 자세 설정
        self._set_initial_pose()

        # VR 브릿지 클라이언트
        self.bridge_client = UnifiedBridgeClient(BRIDGE_ADDR)

        # 실물 로봇 상태
        self.hardware_connected = False
        self.last_hardware_send = time.time()
        self.hardware_send_interval = 0.05  # 20Hz

        # 성능 통계
        self.frame_times = deque(maxlen=240)
        self.last_print = time.time()
        self.frames = 0

        print("✅ Digital Twin 준비 완료\n")

    def setup_dynamixel(self):
        """Dynamixel 초기화"""
        try:
            self.port_handler = PortHandler(DEVICENAME)
            self.packet_handler = PacketHandler(PROTOCOL_VERSION)

            if not self.port_handler.openPort():
                print(f"⚠️ 포트 열기 실패: {DEVICENAME}")
                self.hardware_connected = False
                return

            if not self.port_handler.setBaudRate(BAUDRATE):
                print("⚠️ Baudrate 설정 실패")
                self.hardware_connected = False
                return

            print(f"✅ Dynamixel 연결: {DEVICENAME}")

            # 양팔 토크 ON
            print("🔋 양팔 토크 ON...")
            for motor_id in LEFT_ARM_IDS + RIGHT_ARM_IDS:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
                )
                if result == COMM_SUCCESS:
                    print(f"   모터 {motor_id}: 토크 ON ✓")

            self.hardware_connected = True

        except Exception as e:
            print(f"⚠️ Dynamixel 초기화 실패: {e}")
            self.hardware_connected = False

    def radian_to_value(self, radian):
        """라디안을 Dynamixel 값으로 변환"""
        return int((radian / 0.00153398078) + 2048)

    def send_to_hardware(self):
        """실물 로봇으로 명령 전송"""
        if not self.hardware_connected:
            return

        current_time = time.time()
        if current_time - self.last_hardware_send < self.hardware_send_interval:
            return

        try:
            # 왼팔 전송
            for i, motor_id in enumerate(LEFT_ARM_IDS[:4]):  # 조인트만
                # MuJoCo 값 + 오프셋 = 실물 값
                mujoco_value = self.data.ctrl[self.left_map[f'j{i+1}']]
                hardware_value = mujoco_value + self.left_offsets[i]
                goal_position = self.radian_to_value(hardware_value)

                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, ADDR_GOAL_POSITION, goal_position
                )

            # 왼팔 그리퍼
            gripper_value = self.data.ctrl[self.left_map['g']]
            gripper_position = self.radian_to_value(gripper_value)
            self.packet_handler.write4ByteTxRx(
                self.port_handler, LEFT_ARM_IDS[4], ADDR_GOAL_POSITION, gripper_position
            )

            # 오른팔 전송
            for i, motor_id in enumerate(RIGHT_ARM_IDS[:4]):  # 조인트만
                # MuJoCo 값 + 오프셋 = 실물 값
                mujoco_value = self.data.ctrl[self.right_map[f'j{i+1}']]
                hardware_value = mujoco_value + self.right_offsets[i]
                goal_position = self.radian_to_value(hardware_value)

                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, ADDR_GOAL_POSITION, goal_position
                )

            # 오른팔 그리퍼
            gripper_value = self.data.ctrl[self.right_map['g']]
            gripper_position = self.radian_to_value(gripper_value)
            self.packet_handler.write4ByteTxRx(
                self.port_handler, RIGHT_ARM_IDS[4], ADDR_GOAL_POSITION, gripper_position
            )

            self.last_hardware_send = current_time

        except Exception as e:
            print(f"⚠️ 하드웨어 전송 오류: {e}")

    def _map_actuators(self, side="L"):
        """액추에이터 매핑"""
        if side == "L":
            names = {
                "j1": "actuator_joint1",
                "j2": "actuator_joint2",
                "j3": "actuator_joint3",
                "j4": "actuator_joint4",
                "g": "actuator_gripper_joint",
            }
        else:
            names = {
                "j1": "actuator_joint1_r",
                "j2": "actuator_joint2_r",
                "j3": "actuator_joint3_r",
                "j4": "actuator_joint4_r",
                "g": "actuator_gripper_joint_r",
            }

        out = {}
        for k, nm in names.items():
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except Exception:
                aid = -1
            out[k] = aid
        return out

    def _set_initial_pose(self):
        """초기 자세 설정"""
        init = [0.0, -0.3, 0.8, 0.0]

        def apply(m):
            for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
                aid = m[k]
                if aid < 0:
                    continue
                lo, hi = JOINT_LIMITS[k]
                self.data.ctrl[aid] = float(np.clip(init[i], lo, hi))
            if m['g'] >= 0:
                self.data.ctrl[m['g']] = GRIPPER_RANGE[0]

        apply(self.left_map)
        apply(self.right_map)

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def _apply_packet(self, pkt, mapping, side_name=""):
        """VR 패킷 데이터를 액추에이터에 적용"""
        if not pkt:
            return

        if 'joint_angles' in pkt:
            ja = pkt['joint_angles'][:4]
            for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
                aid = mapping[k]
                if aid < 0:
                    continue
                lo, hi = JOINT_LIMITS[k]
                v = float(np.clip(ja[i], lo, hi))

                if not (np.isnan(v) or np.isinf(v)):
                    self.data.ctrl[aid] = v

        if 'gripper' in pkt and mapping['g'] >= 0:
            gv = float(np.clip(pkt['gripper'], *GRIPPER_RANGE))
            if not (np.isnan(gv) or np.isinf(gv)):
                self.data.ctrl[mapping['g']] = gv

    def _print_status(self):
        """상태 출력"""
        now = time.time()
        if now - self.last_print < 2.0:
            return

        if self.frame_times:
            fps = 1.0 / max(sum(self.frame_times) / len(self.frame_times), 1e-3)
        else:
            fps = 0.0

        print(f"\n📊 FPS {fps:5.1f} | VR Bridge: {self.bridge_client.connected} | Hardware: {self.hardware_connected}")

        if self.hardware_connected:
            print(f"   실물 로봇 제어 중... (20Hz)")

        self.last_print = now

    def run(self):
        """메인 루프"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 카메라 설정
            viewer.cam.distance = CAMERA_DISTANCE
            viewer.cam.azimuth = AZIMUTH_FRONT
            viewer.cam.elevation = CAMERA_ELEVATION

            print("\n✨ === Digital Twin System ===")
            print("📡 VR 입력 대기 중...")
            print("🤖 양팔 동기화 활성화")
            print("🎮 Meta Quest 2로 제어하세요")
            print("Press ESC to exit\n")

            while viewer.is_running():
                t0 = time.time()

                # VR 브릿지에서 데이터 받기
                left_packet = self.bridge_client.pop_latest_left()
                right_packet = self.bridge_client.pop_latest_right()

                # MuJoCo에 적용
                self._apply_packet(left_packet, self.left_map, "LEFT")
                self._apply_packet(right_packet, self.right_map, "RIGHT")

                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)

                # 실물 로봇으로 전송 (20Hz)
                self.send_to_hardware()

                # 화면 업데이트
                viewer.sync()

                dt = time.time() - t0
                self.frame_times.append(dt)
                self.frames += 1
                self._print_status()
                time.sleep(max(0.0, 0.008 - dt))

        print("🏁 Digital Twin 종료")

    def cleanup(self):
        """종료 처리"""
        if self.hardware_connected and hasattr(self, 'port_handler'):
            # 토크 OFF
            print("🔓 양팔 토크 OFF...")
            for motor_id in LEFT_ARM_IDS + RIGHT_ARM_IDS:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
                )
            self.port_handler.closePort()
            print("✅ Dynamixel 포트 닫음")

if __name__ == "__main__":
    try:
        print("🚀 Starting Digital Twin System")
        controller = DualArmDigitalTwin()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        import traceback
        print(f"❌ 오류: {e}")
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.cleanup()
        print("🏁 종료")