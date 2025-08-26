#!/usr/bin/env python3
"""
양팔 제어 수정 버전
scene_dual.xml 사용
"""

import socket
import json
import time
import threading
from collections import deque
import numpy as np
import mujoco
import mujoco.viewer

# 경로/소켓 설정
XML_SCENE_PATH = 'scene_dual.xml'  # 이제 수정된 omx_r.xml과 함께 작동!
BRIDGE_ADDR = ('localhost', 12345)

# 카메라 설정
CAMERA_MODE = 'behind'
CAMERA_BIAS_Y = 0.0
CAMERA_LIFT_Z = 0.25
CAMERA_DISTANCE = 2.0
AZIMUTH_FRONT = 180  # 180도로 변경하여 뒤에서 보기
AZIMUTH_BEHIND = (AZIMUTH_FRONT + 180) % 360
CAMERA_ELEVATION = -15

# 조인트 안전 범위
JOINT_LIMITS = {
    'j1': (-3.14, 3.14),
    'j2': (-1.5, 1.5),
    'j3': (-1.5, 1.4),
    'j4': (-1.7, 1.97)
}
GRIPPER_RANGE = (-0.01, 0.019)
PRINT_PERIOD_S = 2.0

def _get_body_xpos(model, data, candidates):
    """바디 위치 찾기 헬퍼"""
    for name in candidates:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return data.xpos[bid]
        except Exception:
            pass
    return np.array([0.0, 0.0, 0.0])

def center_cam_on_arms(model, data, viewer, *, bias_y=0.0, lift=0.18,
                       distance=1.7, azimuth_front=0, elevation=-15, mode='front'):
    """양팔 중앙에 카메라 포커스"""
    mujoco.mj_forward(model, data)
    pL = _get_body_xpos(model, data, ["link2", "arm_base_l", "base"])
    pR = _get_body_xpos(model, data, ["link2_r", "arm_base_r", "base_r"])
    center = 0.5 * (pL + pR)
    center[1] += bias_y
    
    az = (azimuth_front + 180) % 360 if mode == 'behind' else azimuth_front
    
    viewer.cam.lookat[:] = [float(center[0]), float(center[1]), float(center[2] + lift)]
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
        self.last_data_time = time.time()
        self.recv_intervals = deque(maxlen=120)
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
                except Exception:
                    pass
                self.connected = True
                print(f"🔗 Connected to dual_arm_bridge at {self.addr}")

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
                                now = time.time()
                                self.recv_intervals.append(now - self.last_data_time)
                                self.last_data_time = now
                                
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
                    print(f"⚠️  Connection lost: {e}")
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

    def hz(self):
        if not self.recv_intervals:
            return 0.0
        avg = sum(self.recv_intervals) / len(self.recv_intervals)
        return 1.0 / max(avg, 1e-3)

class DualMuJoCoController:
    """듀얼 무조코 컨트롤러"""
    def __init__(self):
        print("🎯 Loading:", XML_SCENE_PATH)
        self.model = mujoco.MjModel.from_xml_path(XML_SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        
        # 액추에이터 매핑 디버그
        print("\n=== 전체 액추에이터 목록 ===")
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i}: {name}")
        
        self.left_map = self._map_actuators(side="L")
        self.right_map = self._map_actuators(side="R")
        
        self._set_initial_pose()
        
        self.bridge_client = UnifiedBridgeClient(BRIDGE_ADDR)
        
        self.frame_times = deque(maxlen=240)
        self.last_print = time.time()
        self.frames = 0

    def _map_actuators(self, side="L"):
        """액추에이터 매핑"""
        if side == "L":
            # 왼팔 - omx.xml의 액추에이터 이름
            names = {
                "j1": "actuator_joint1",
                "j2": "actuator_joint2", 
                "j3": "actuator_joint3",
                "j4": "actuator_joint4",
                "g": "actuator_gripper_joint",
            }
        else:
            # 오른팔 - omx_r.xml의 액추에이터 이름  
            names = {
                "j1": "actuator_joint1_r",
                "j2": "actuator_joint2_r",
                "j3": "actuator_joint3_r", 
                "j4": "actuator_joint4_r",
                "g": "actuator_gripper_joint_r",
            }
        
        out = {}
        print(f"\n🔧 Mapping {side} arm:")
        for k, nm in names.items():
            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            except Exception:
                aid = -1
            out[k] = aid
            status = "✅" if aid >= 0 else "❌"
            print(f"  {status} {k} -> {nm} (id={aid})")
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
        print("✅ Initial pose set")

    def _apply_packet(self, pkt, mapping, side_name=""):
        """패킷 데이터를 액추에이터에 적용"""
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
                
                # 오른팔 Joint1 반전 제거 (test3_dual.py와 동일하게)
                # if side_name == "RIGHT" and k == 'j1':
                #     v = -v
                    
                if not (np.isnan(v) or np.isinf(v)):
                    self.data.ctrl[aid] = v
        
        if 'gripper' in pkt and mapping['g'] >= 0:
            gv = float(np.clip(pkt['gripper'], *GRIPPER_RANGE))
            if not (np.isnan(gv) or np.isinf(gv)):
                self.data.ctrl[mapping['g']] = gv

    def _print_status(self):
        """상태 출력"""
        now = time.time()
        if now - self.last_print < PRINT_PERIOD_S:
            return
        
        if self.frame_times:
            fps = 1.0 / max(sum(self.frame_times) / len(self.frame_times), 1e-3)
        else:
            fps = 0.0
        
        print(f"\n📊 FPS {fps:5.1f} | Bridge {self.bridge_client.hz():5.1f} Hz | Connected: {self.bridge_client.connected}")
        
        # 액추에이터 값 확인
        if self.left_map['j1'] >= 0 and self.right_map['j1'] >= 0:
            left_j1 = self.data.ctrl[self.left_map['j1']]
            right_j1 = self.data.ctrl[self.right_map['j1']]
            print(f"   Joint1 values: L={left_j1:.3f}, R={right_j1:.3f}")
        
        self.last_print = now

    def run(self):
        """메인 루프"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            center_cam_on_arms(
                self.model, self.data, viewer,
                bias_y=CAMERA_BIAS_Y, lift=CAMERA_LIFT_Z,
                distance=CAMERA_DISTANCE,
                azimuth_front=AZIMUTH_FRONT,
                elevation=CAMERA_ELEVATION,
                mode=CAMERA_MODE
            )
            
            print("\n✨ === MuJoCo Dual Arm Controller ===")
            print("📡 Waiting for data from dual_arm_bridge...")
            print("🎮 Control both arms with Meta Quest 2 controllers")
            print("Press ESC to exit\n")
            
            while viewer.is_running():
                t0 = time.time()
                
                # 브리지에서 데이터 받기
                left_packet = self.bridge_client.pop_latest_left()
                right_packet = self.bridge_client.pop_latest_right()
                
                # 액추에이터에 적용
                self._apply_packet(left_packet, self.left_map, "LEFT")
                self._apply_packet(right_packet, self.right_map, "RIGHT")
                
                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 카메라 주기적 업데이트
                if self.frames % 120 == 0:
                    center_cam_on_arms(
                        self.model, self.data, viewer,
                        bias_y=CAMERA_BIAS_Y, lift=CAMERA_LIFT_Z,
                        distance=CAMERA_DISTANCE,
                        azimuth_front=AZIMUTH_FRONT,
                        elevation=CAMERA_ELEVATION,
                        mode=CAMERA_MODE
                    )
                
                dt = time.time() - t0
                self.frame_times.append(dt)
                self.frames += 1
                self._print_status()
                time.sleep(max(0.0, 0.008 - dt))
        
        print("🏁 Controller closed")

if __name__ == "__main__":
    try:
        print("🚀 Starting MuJoCo dual arm controller")
        DualMuJoCoController().run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        print("🏁 Exit")
