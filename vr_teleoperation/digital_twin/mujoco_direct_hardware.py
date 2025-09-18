#!/usr/bin/env python3
"""
MuJoCo Direct Hardware Control - VR Teleoperation with Hardware Direct Control
VR → Bridge → MuJoCo (Visualization + Hardware Control)
오프셋 보정값 적용된 직접 하드웨어 제어
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

# 경로 설정
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

# 양팔 모터 ID
LEFT_ARM_IDS = [11, 12, 13, 14]
LEFT_GRIPPER_ID = 15
RIGHT_ARM_IDS = [21, 22, 23, 24]
RIGHT_GRIPPER_ID = 25

# 오프셋 보정값 (teaching_1.py에서 가져옴)
LEFT_OFFSETS = [0.0, -0.43, 1.94, -0.42]
RIGHT_OFFSETS = [0.66, -1.03, 0.96, -2.07]

# 카메라 설정
CAMERA_MODE = 'behind'
CAMERA_DISTANCE = 2.0
CAMERA_ELEVATION = -15
AZIMUTH_FRONT = 180

# 조인트 안전 범위
JOINT_LIMITS = {
    'j1': (-3.14, 3.14),
    'j2': (-1.5, 1.5),
    'j3': (-1.5, 1.4),
    'j4': (-1.7, 1.97)
}
GRIPPER_RANGE = (-0.01, 0.019)

def _get_body_xpos(model, data, candidates):
    """바디 위치 찾기"""
    for name in candidates:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return data.xpos[bid]
        except:
            pass
    return np.array([0.0, 0.0, 0.0])

def center_cam_on_arms(model, data, viewer, distance=2.0, elevation=-15, mode='behind'):
    """양팔 중앙에 카메라 포커스"""
    mujoco.mj_forward(model, data)
    pL = _get_body_xpos(model, data, ["link2", "arm_base_l", "base"])
    pR = _get_body_xpos(model, data, ["link2_r", "arm_base_r", "base_r"])
    center = 0.5 * (pL + pR)

    az = 0 if mode == 'behind' else 180

    viewer.cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2] + 0.25)]
    viewer.cam.distance = distance
    viewer.cam.azimuth = az
    viewer.cam.elevation = elevation

class UnifiedBridgeClient:
    """통합 브리지 클라이언트"""
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
                try:
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except:
                    pass
                self.connected = True
                print(f"🔗 Bridge 연결: {self.addr}")

                while True:
                    try:
                        raw = self.sock.recv(8192).decode('utf-8', errors='ignore')
                        if not raw:
                            raise ConnectionError("연결 끊김")
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
                    print(f"⚠️ 연결 끊김: {e}")
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

class DirectHardwareController:
    """MuJoCo 시각화 + 직접 하드웨어 제어"""
    def __init__(self):
        print("🎯 MuJoCo Direct Hardware Controller 시작")
        print("📊 오프셋 보정값 적용:")
        print(f"   왼팔:  {LEFT_OFFSETS}")
        print(f"   오른팔: {RIGHT_OFFSETS}\n")

        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(XML_SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        # 액추에이터 매핑
        self.left_map = self._map_actuators(side="L")
        self.right_map = self._map_actuators(side="R")

        # Dynamixel 초기화
        self.setup_dynamixel()

        # 브리지 클라이언트
        self.bridge_client = UnifiedBridgeClient(BRIDGE_ADDR)

        # 현재 조인트 값
        self.left_joints = [0.0, 0.0, 0.0, 0.0]
        self.right_joints = [0.0, 0.0, 0.0, 0.0]
        self.left_gripper = -0.01
        self.right_gripper = -0.01

        # 스무딩을 위한 히스토리
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)

        # 통계
        self.frame_times = deque(maxlen=240)
        self.last_print = time.time()
        self.frames = 0
        self.hardware_update_count = 0

        # 초기 자세
        self._set_initial_pose()

    def setup_dynamixel(self):
        """Dynamixel 초기화"""
        try:
            self.port_handler = PortHandler(DEVICENAME)
            self.packet_handler = PacketHandler(PROTOCOL_VERSION)

            if not self.port_handler.openPort():
                print(f"⚠️ {DEVICENAME} 열기 실패, /dev/ttyUSB0 시도...")
                self.port_handler = PortHandler('/dev/ttyUSB0')
                if not self.port_handler.openPort():
                    print("❌ Dynamixel 포트 열기 실패")
                    self.hardware_enabled = False
                    return

            if not self.port_handler.setBaudRate(BAUDRATE):
                print("❌ Baudrate 설정 실패")
                self.hardware_enabled = False
                return

            print(f"✅ Dynamixel 연결: {self.port_handler.port_name}")

            # 모든 모터 토크 ON
            all_motors = (LEFT_ARM_IDS + [LEFT_GRIPPER_ID] +
                         RIGHT_ARM_IDS + [RIGHT_GRIPPER_ID])

            for motor_id in all_motors:
                result, error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
                )
                if result == COMM_SUCCESS:
                    print(f"   모터 {motor_id}: 토크 ON ✓")

            self.hardware_enabled = True
            print("✅ 하드웨어 준비 완료\n")

        except Exception as e:
            print(f"❌ Dynamixel 초기화 오류: {e}")
            self.hardware_enabled = False

    def _map_actuators(self, side="L"):
        """MuJoCo 액추에이터 매핑"""
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
        print(f"🔧 {side}팔 액추에이터 매핑:")
        for k, nm in names.items():
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except:
                aid = -1
            out[k] = aid
            print(f"  {'✅' if aid >= 0 else '❌'} {k} -> {nm} (id={aid})")
        return out

    def _set_initial_pose(self):
        """초기 자세 설정"""
        init = [0.0, -0.3, 0.8, 0.0]

        # MuJoCo 초기 자세
        for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
            if self.left_map[k] >= 0:
                self.data.ctrl[self.left_map[k]] = init[i]
            if self.right_map[k] >= 0:
                self.data.ctrl[self.right_map[k]] = init[i]

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        # 하드웨어 초기 자세
        if self.hardware_enabled:
            self.send_to_hardware(init, init, -0.01, -0.01)

        print("✅ 초기 자세 설정 완료")

    def radian_to_value(self, radian):
        """라디안을 Dynamixel 값으로 변환"""
        return int(radian / 0.00153398078 + 2048)

    def send_to_hardware(self, left_joints, right_joints, left_gripper, right_gripper):
        """하드웨어로 직접 전송 (오프셋 적용)"""
        if not self.hardware_enabled:
            return

        try:
            # 왼팔 전송 (오프셋 적용)
            for i, motor_id in enumerate(LEFT_ARM_IDS):
                goal_rad = left_joints[i] + LEFT_OFFSETS[i]
                goal_value = self.radian_to_value(goal_rad)
                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, ADDR_GOAL_POSITION, goal_value
                )

            # 왼팔 그리퍼
            gripper_value = self.radian_to_value(left_gripper)
            self.packet_handler.write4ByteTxRx(
                self.port_handler, LEFT_GRIPPER_ID, ADDR_GOAL_POSITION, gripper_value
            )

            # 오른팔 전송 (오프셋 적용)
            for i, motor_id in enumerate(RIGHT_ARM_IDS):
                goal_rad = right_joints[i] + RIGHT_OFFSETS[i]
                goal_value = self.radian_to_value(goal_rad)
                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, ADDR_GOAL_POSITION, goal_value
                )

            # 오른팔 그리퍼
            gripper_value = self.radian_to_value(right_gripper)
            self.packet_handler.write4ByteTxRx(
                self.port_handler, RIGHT_GRIPPER_ID, ADDR_GOAL_POSITION, gripper_value
            )

            self.hardware_update_count += 1

        except Exception as e:
            if self.frames % 600 == 0:  # 5초마다 에러 출력
                print(f"⚠️ 하드웨어 전송 오류: {e}")

    def smooth_joints(self, new_joints, history):
        """조인트 값 스무딩"""
        history.append(new_joints)
        if len(history) < 3:
            return new_joints

        # 가중 평균 (최신 값에 더 높은 가중치)
        weights = np.array([0.2, 0.3, 0.5])[-len(history):]
        weights = weights / weights.sum()

        smoothed = np.average(list(history), axis=0, weights=weights)
        return smoothed.tolist()

    def _apply_packet(self, pkt, mapping, side_name=""):
        """패킷 데이터 적용 및 하드웨어 업데이트"""
        if not pkt:
            return None, None

        joints = None
        gripper = None

        if 'joint_angles' in pkt:
            ja = pkt['joint_angles'][:4]
            joints = []
            for i, k in enumerate(['j1', 'j2', 'j3', 'j4']):
                aid = mapping[k]
                lo, hi = JOINT_LIMITS[k]
                v = float(np.clip(ja[i], lo, hi))
                joints.append(v)

                if aid >= 0 and not (np.isnan(v) or np.isinf(v)):
                    self.data.ctrl[aid] = v

        if 'gripper' in pkt and mapping['g'] >= 0:
            gripper = float(np.clip(pkt['gripper'], *GRIPPER_RANGE))
            if not (np.isnan(gripper) or np.isinf(gripper)):
                self.data.ctrl[mapping['g']] = gripper

        return joints, gripper

    def _print_status(self):
        """상태 출력"""
        now = time.time()
        if now - self.last_print < 2.0:
            return

        if self.frame_times:
            fps = 1.0 / max(sum(self.frame_times) / len(self.frame_times), 1e-3)
        else:
            fps = 0.0

        print(f"\n📊 === 상태 ===")
        print(f"  MuJoCo FPS: {fps:.1f}")
        print(f"  Bridge 연결: {'✅' if self.bridge_client.connected else '❌'}")
        print(f"  하드웨어: {'✅ 활성' if self.hardware_enabled else '❌ 비활성'}")

        if self.hardware_enabled:
            print(f"  하드웨어 업데이트: {self.hardware_update_count}회")

            # 현재 조인트 값 (오프셋 적용 전)
            left_str = ', '.join([f'{j:.2f}' for j in self.left_joints])
            right_str = ', '.join([f'{j:.2f}' for j in self.right_joints])
            print(f"  왼팔:  [{left_str}]")
            print(f"  오른팔: [{right_str}]")

        self.last_print = now

    def run(self):
        """메인 루프"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            center_cam_on_arms(
                self.model, self.data, viewer,
                distance=CAMERA_DISTANCE,
                elevation=CAMERA_ELEVATION,
                mode=CAMERA_MODE
            )

            print("\n✨ === MuJoCo Direct Hardware Control ===")
            print("🎮 VR로 양팔 제어 중...")
            print("📡 MuJoCo 시각화 + 실물 하드웨어 직접 제어")
            print("ESC로 종료\n")

            # 하드웨어 업데이트 주기 제어
            last_hw_update = time.time()
            hw_update_interval = 0.02  # 50Hz 하드웨어 업데이트

            while viewer.is_running():
                t0 = time.time()

                # 브리지에서 데이터 받기
                left_packet = self.bridge_client.pop_latest_left()
                right_packet = self.bridge_client.pop_latest_right()

                # MuJoCo에 적용 및 조인트 값 추출
                left_data = self._apply_packet(left_packet, self.left_map, "LEFT")
                right_data = self._apply_packet(right_packet, self.right_map, "RIGHT")

                # 새 데이터가 있으면 업데이트
                if left_data[0] is not None:
                    self.left_joints = self.smooth_joints(left_data[0], self.left_history)
                if left_data[1] is not None:
                    self.left_gripper = left_data[1]

                if right_data[0] is not None:
                    self.right_joints = self.smooth_joints(right_data[0], self.right_history)
                if right_data[1] is not None:
                    self.right_gripper = right_data[1]

                # 하드웨어 업데이트 (주기적으로)
                current_time = time.time()
                if current_time - last_hw_update >= hw_update_interval:
                    self.send_to_hardware(
                        self.left_joints, self.right_joints,
                        self.left_gripper, self.right_gripper
                    )
                    last_hw_update = current_time

                # MuJoCo 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 카메라 주기적 업데이트
                if self.frames % 120 == 0:
                    center_cam_on_arms(
                        self.model, self.data, viewer,
                        distance=CAMERA_DISTANCE,
                        elevation=CAMERA_ELEVATION,
                        mode=CAMERA_MODE
                    )

                dt = time.time() - t0
                self.frame_times.append(dt)
                self.frames += 1
                self._print_status()

                # FPS 제어
                time.sleep(max(0.0, 0.008 - dt))

        print("🏁 종료 중...")
        self.cleanup()

    def cleanup(self):
        """종료 처리"""
        if self.hardware_enabled and hasattr(self, 'port_handler'):
            # 모터 토크 OFF
            all_motors = (LEFT_ARM_IDS + [LEFT_GRIPPER_ID] +
                         RIGHT_ARM_IDS + [RIGHT_GRIPPER_ID])

            print("\n🔓 모터 토크 해제 중...")
            for motor_id in all_motors:
                try:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
                    )
                    print(f"   모터 {motor_id}: 토크 OFF ✓")
                except:
                    pass

            self.port_handler.closePort()
            print("✅ Dynamixel 포트 닫음")

if __name__ == "__main__":
    try:
        print("🚀 MuJoCo Direct Hardware Controller 시작")
        controller = DirectHardwareController()
        controller.run()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        import traceback
        print(f"❌ 오류: {e}")
        traceback.print_exc()
    finally:
        print("🏁 프로그램 종료")