#!/usr/bin/env python3
"""
🎮 Meta Quest 2 VR → MuJoCo Bridge (Docker/ROS1)

이 파일은 Meta Quest 2 VR 헤드셋의 컨트롤러 포즈를 OpenManipulator-X 
로봇의 조인트 각도로 변환하는 핵심 VR 브릿지입니다.

주요 특징:
🎯 Orientation 기반 Joint4 직관적 제어 강화
- Pitch(Y축 회전) → Joint4 직접 매핑 강화  
- Roll/Yaw 보조 제어 추가
- 손목 꺾기 동작 정밀 반영

🔄 Offset-based Control Method:
- 절대 좌표 대신 상대적 움직임으로 제어
- VR 공간과 로봇 작업공간 불일치 문제 해결
- 안전하고 직관적인 로봇 제어

📡 통신:
- ROS1 노드로 VR 데이터 수신
- Socket을 통해 Host(mirror1.py)로 조인트 값 전송
- 120Hz 고속 제어 루프

⚙️ 매핑 방식:
- Position → Joint1,2,3 (어깨, 팔꿈치, 손목)
- Orientation → Joint4 (손목 회전)
- 스무딩 필터 적용으로 안정성 확보
"""

import rospy
import numpy as np
import json
import socket
import threading
import time
from collections import deque
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import tf.transformations as tf_trans
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt

class IntuitiveJoint4VRBridge:
    def __init__(self):
        rospy.init_node('intuitive_joint4_vr_bridge')
        
        print("🎯 Joint4 직관적 제어 강화 VR Bridge V3.2 시작")
        print("🤏 손목 회전을 Joint4로 정밀 매핑")
        
        # 소켓 서버 설정
        self.setup_socket_server()
        
        # VR 데이터 저장
        self.vr_data = {
            'hand_pose': {
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'initial_position': None,
                'initial_orientation': None,
                'calibrated': False,
                'pose_history': deque(maxlen=15),
                'orientation_history': deque(maxlen=20)  # orientation 전용 히스토리
            },
            'inputs': {
                'trigger': 0.0,
                'button_upper': False,
                'button_lower': False
            }
        }
        
        # 로봇 조인트 상태
        self.robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.target_joints = [0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0]
        self.gripper_value = -0.01
        
        # 안전 파라미터 (V3.1 유지)
        self.safety_params = {
            'max_joint_speed': 0.08,
            'position_scale': np.array([0.7, 0.7, 0.7]),
            'smooth_factor': 0.08,
            'z_axis_gain': 0.4,
            'joint1_gain': 0.55,
            'safety_margin': 0.9
        }
        
        # Z축 매핑 (V3.1 유지)
        self.z_axis_mapping = {
            'down_threshold': -0.08,
            'up_threshold': 0.08,
            'joint2_down_gain': 0.35,
            'joint3_down_gain': 0.25,
            'joint2_up_gain': -0.25,
            'joint3_up_gain': -0.35,
        }
        
        # 🎯 Joint4 직관적 매핑 (강화)
        self.joint4_mapping = {
            'pitch_sensitivity': 1.2,      # 크게 증가 (0.4 → 1.2)
            'roll_sensitivity': 0.3,        # Roll 영향 추가
            'yaw_sensitivity': 0.2,         # Yaw 영향 추가
            'direct_influence': 0.7,        # 직접 영향도 증가 (0.3 → 0.7)
            'smoothing_factor': 0.15,       # Joint4 전용 스무딩 감소 (더 반응적)
            'deadzone': 0.05,              # 데드존 설정
            'max_change_rate': 0.12        # Joint4 최대 변화율
        }
        
        # Orientation 분석 데이터 (실제 데이터 기반)
        self.orientation_patterns = {
            'pitch_down': {'range': [0.5, 1.5], 'joint4_target': 0.8},    # 아래로 꺾기
            'pitch_up': {'range': [-1.5, -0.5], 'joint4_target': -0.8},   # 위로 꺾기
            'neutral': {'range': [-0.3, 0.3], 'joint4_target': 0.0},      # 중립
            'roll_left': {'range': [0.3, 1.0], 'joint4_modifier': 0.2},   # 좌측 기울기
            'roll_right': {'range': [-1.0, -0.3], 'joint4_modifier': -0.2} # 우측 기울기
        }
        
        # 실제 수집 데이터 로드
        self.load_orientation_enhanced_data()
        
        # 다중 KD-Tree 구성
        self.build_orientation_trees()
        
        # 떨림 제거 필터
        self.setup_joint4_filter()
        
        # 성능 통계
        self.stats = {
            'orientation_direct_mapping': 0,
            'position_mapping': 0,
            'control_frequency': 0.0,
            'joint4_response_quality': 0.0
        }
        
        # ROS 설정
        self.setup_ros_topics()
        
        # 제어 루프 시작
        self.control_thread = threading.Thread(target=self.orientation_enhanced_control_loop, daemon=True)
        self.control_thread.start()
        
        # 디버그 스레드
        self.debug_thread = threading.Thread(target=self.debug_loop, daemon=True)
        self.debug_thread.start()
        
        print("✅ Joint4 직관적 제어 강화 시스템 준비 완료")
        print("🤏 손목 꺾기 → Joint4 정밀 반영")
    
    def setup_joint4_filter(self):
        """Joint4 전용 필터 (더 반응적)"""
        # 일반 조인트용 필터
        self.filter_freq = 5.0
        self.filter_order = 3
        
        # Joint4 전용 필터 (더 높은 주파수)
        self.joint4_filter_freq = 10.0  # 더 빠른 반응
        self.joint4_filter_order = 2
        
        nyquist = 120.0 / 2
        
        # 일반 필터
        normalized_freq = self.filter_freq / nyquist
        self.filter_b, self.filter_a = butter(self.filter_order, normalized_freq, btype='low')
        
        # Joint4 필터
        j4_normalized_freq = self.joint4_filter_freq / nyquist
        self.j4_filter_b, self.j4_filter_a = butter(self.joint4_filter_order, j4_normalized_freq, btype='low')
        
        self.filter_history = {
            'joint_targets': deque(maxlen=30),
            'joint4_targets': deque(maxlen=15),  # Joint4 전용
            'vr_deltas': deque(maxlen=15)
        }
    
    def load_orientation_enhanced_data(self):
        """Orientation 변화를 강조한 데이터 로드"""
        print("📁 Orientation 강화 매핑 데이터 로드 중...")
        
        # 기본 데이터 (V3.1과 동일)
        base_data = [
            {"vr_pos": [0.001, -0.013, -0.0003], "vr_ori": [-0.022, 0.043, -0.054], "joints": [0.0, 0.0, 0.0, 0.0]},
            {"vr_pos": [0.019, -0.003, -0.021], "vr_ori": [0.149, 0.689, 0.109], "joints": [0.0, 0.0, 0.0, 0.5]},  # pitch up
            {"vr_pos": [-0.036, -0.012, 0.048], "vr_ori": [-0.037, -0.672, -0.056], "joints": [0.0, 0.0, 0.0, -0.5]}, # pitch down
        ]
        
        # Orientation 중심 증강 데이터
        orientation_samples = []
        
        # 1. Pitch 변화 샘플 (손목 위아래)
        for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]:
            for pitch in np.linspace(-1.2, 1.2, 9):  # 더 세밀한 pitch 범위
                # Pitch에 따른 Joint4 직접 매핑
                joint4 = pitch * 0.7  # 강한 상관관계
                sample = {
                    "vr_pos": pos,
                    "vr_ori": [0.0, pitch, 0.0],  # Pitch 중심
                    "joints": [0.0, 0.0, 0.0, joint4]
                }
                orientation_samples.append(sample)
        
        # 2. Roll 변화 샘플 (손목 좌우 기울기)
        for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]:
            for roll in [-0.8, -0.4, 0.0, 0.4, 0.8]:
                # Roll은 Joint4에 보조적 영향
                joint4 = roll * 0.3
                sample = {
                    "vr_pos": pos,
                    "vr_ori": [roll, 0.0, 0.0],  # Roll 중심
                    "joints": [0.0, 0.0, 0.0, joint4]
                }
                orientation_samples.append(sample)
        
        # 3. 복합 orientation 샘플 (실제 손목 동작)
        complex_orientations = [
            # 아래로 꺾으면서 좌측 기울기
            {"ori": [0.3, 0.8, 0.1], "j4": 0.7},
            # 위로 꺾으면서 우측 기울기
            {"ori": [-0.3, -0.8, -0.1], "j4": -0.7},
            # 중립에서 좌우 회전
            {"ori": [0.5, 0.0, 0.5], "j4": 0.3},
            {"ori": [-0.5, 0.0, -0.5], "j4": -0.3},
            # 극단적 꺾기
            {"ori": [0.0, 1.2, 0.0], "j4": 0.9},
            {"ori": [0.0, -1.2, 0.0], "j4": -0.9},
        ]
        
        for comp in complex_orientations:
            for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, -0.1]]:
                sample = {
                    "vr_pos": pos,
                    "vr_ori": comp["ori"],
                    "joints": [0.0, 0.0, 0.0, comp["j4"]]
                }
                orientation_samples.append(sample)
        
        # 4. 위치별 orientation 변화 (중요!)
        position_orientation_pairs = [
            # 앞으로 뻗으면서 아래로 꺾기
            {"pos": [0.15, 0.0, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [0.0, 0.3, -0.3, 0.6]},
            # 앞으로 뻗으면서 위로 꺾기
            {"pos": [0.15, 0.0, 0.0], "ori": [0.0, -0.8, 0.0], "joints": [0.0, 0.3, -0.3, -0.6]},
            # 옆으로 뻗으면서 꺾기
            {"pos": [0.0, 0.15, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [0.4, 0.0, 0.0, 0.6]},
            {"pos": [0.0, -0.15, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [-0.4, 0.0, 0.0, 0.6]},
            # 아래로 내리면서 꺾기
            {"pos": [0.0, 0.0, -0.15], "ori": [0.0, 0.8, 0.0], "joints": [0.0, 0.3, 0.2, 0.6]},
            {"pos": [0.0, 0.0, -0.15], "ori": [0.0, -0.8, 0.0], "joints": [0.0, 0.3, 0.2, -0.6]},
        ]
        
        for pair in position_orientation_pairs:
            orientation_samples.append({
                "vr_pos": pair["pos"],
                "vr_ori": pair["ori"],
                "joints": pair["joints"]
            })
        
        # 모든 데이터 결합
        self.mapping_data = base_data + orientation_samples
        
        print(f"✅ {len(self.mapping_data)}개 Orientation 강화 데이터 로드 완료")
        print(f"   기본: {len(base_data)}개, Orientation 특화: {len(orientation_samples)}개")
    
    def build_orientation_trees(self):
        """Orientation 특화 KD-Tree 구성"""
        print("🔍 Orientation 특화 KD-Tree 구성 중...")
        
        # 1. 위치 기반 트리 (기존)
        positions = np.array([sample['vr_pos'] for sample in self.mapping_data])
        self.position_tree = cKDTree(positions)
        
        # 2. Orientation 전용 트리 (3D)
        orientations = np.array([sample['vr_ori'] for sample in self.mapping_data])
        self.orientation_tree = cKDTree(orientations)
        
        # 3. Pitch 중심 트리 (1D를 3D로 확장)
        pitch_features = []
        for sample in self.mapping_data:
            # Pitch를 중심으로, roll과 yaw는 보조
            pitch_weighted = [
                sample['vr_ori'][0] * 0.3,  # Roll (낮은 가중치)
                sample['vr_ori'][1] * 1.5,  # Pitch (높은 가중치)
                sample['vr_ori'][2] * 0.2   # Yaw (낮은 가중치)
            ]
            pitch_features.append(pitch_weighted)
        self.pitch_tree = cKDTree(np.array(pitch_features))
        
        # 4. 복합 트리 (위치 + Orientation)
        combined_features = []
        for sample in self.mapping_data:
            # 위치와 orientation 결합 (orientation에 더 높은 가중치)
            combined = (
                list(np.array(sample['vr_pos']) * 0.5) +  # 위치 (낮은 가중치)
                list(np.array(sample['vr_ori']) * 1.2)    # Orientation (높은 가중치)
            )
            combined_features.append(combined)
        self.combined_tree = cKDTree(np.array(combined_features))
        
        print(f"🔍 Orientation 특화 트리 구성 완료:")
        print(f"   위치 트리: {len(positions)}개")
        print(f"   Orientation 트리: {len(orientations)}개")
        print(f"   Pitch 중심 트리: {len(pitch_features)}개")
        print(f"   복합 트리: {len(combined_features)}개")
    
    def intuitive_joint4_control(self, vr_pos_delta, vr_ori_delta):
        """직관적 Joint4 제어 알고리즘"""
        # Joint 1-3는 기존 방식 유지
        joint1 = np.arctan2(vr_pos_delta[1], vr_pos_delta[0] + 0.25) * self.safety_params['joint1_gain']
        joint1 = np.clip(joint1, -0.65, 0.65)
        
        # Z축 기반 Joint2, Joint3
        z_delta = vr_pos_delta[2]
        if z_delta < self.z_axis_mapping['down_threshold']:
            z_factor = min(abs(z_delta) / 0.25, 1.0)
            joint2 = z_factor * self.z_axis_mapping['joint2_down_gain']
            joint3 = z_factor * self.z_axis_mapping['joint3_down_gain']
        elif z_delta > self.z_axis_mapping['up_threshold']:
            z_factor = min(z_delta / 0.25, 1.0)
            joint2 = z_factor * self.z_axis_mapping['joint2_up_gain']
            joint3 = z_factor * self.z_axis_mapping['joint3_up_gain']
        else:
            # KD-Tree 보간
            distances, indices = self.position_tree.query(vr_pos_delta, k=min(4, len(self.mapping_data)))
            if isinstance(distances, float):
                distances = [distances]
                indices = [indices]
            weights = 1.0 / (np.array(distances) + 1e-6)
            weights = weights / np.sum(weights)
            joint2 = sum(weights[i] * self.mapping_data[idx]['joints'][1] for i, idx in enumerate(indices))
            joint3 = sum(weights[i] * self.mapping_data[idx]['joints'][2] for i, idx in enumerate(indices))
        
        # 🎯 Joint4: Orientation 기반 직관적 제어 (핵심!)
        # 1. Pitch 기반 직접 매핑 (주요 제어)
        pitch = vr_ori_delta[1]
        roll = vr_ori_delta[0]
        yaw = vr_ori_delta[2]
        
        # 데드존 적용
        if abs(pitch) < self.joint4_mapping['deadzone']:
            pitch = 0.0
        
        # Pitch → Joint4 직접 매핑 (강화)
        joint4_from_pitch = pitch * self.joint4_mapping['pitch_sensitivity']
        
        # Roll과 Yaw 보조 영향
        joint4_from_roll = roll * self.joint4_mapping['roll_sensitivity']
        joint4_from_yaw = yaw * self.joint4_mapping['yaw_sensitivity']
        
        # 2. Orientation KD-Tree 보간
        pitch_weighted_query = [
            roll * 0.3,
            pitch * 1.5,
            yaw * 0.2
        ]
        
        ori_distances, ori_indices = self.pitch_tree.query(pitch_weighted_query, k=min(6, len(self.mapping_data)))
        
        if isinstance(ori_distances, float):
            ori_distances = [ori_distances]
            ori_indices = [ori_indices]
        
        # 거리 기반 가중치
        ori_weights = 1.0 / (np.array(ori_distances) + 1e-6)
        ori_weights = ori_weights / np.sum(ori_weights)
        
        joint4_from_tree = sum(ori_weights[i] * self.mapping_data[idx]['joints'][3] 
                               for i, idx in enumerate(ori_indices))
        
        # 3. 최종 Joint4 결합 (직접 매핑 우선)
        joint4_direct = joint4_from_pitch + joint4_from_roll * 0.3 + joint4_from_yaw * 0.2
        
        joint4 = (
            self.joint4_mapping['direct_influence'] * joint4_direct +
            (1 - self.joint4_mapping['direct_influence']) * joint4_from_tree
        )
        
        # 4. 위치 기반 보정 (옵션)
        # 멀리 뻗을 때 Joint4 반응 증가
        reach = np.sqrt(vr_pos_delta[0]**2 + vr_pos_delta[1]**2)
        if reach > 0.15:
            reach_factor = min((reach - 0.15) / 0.15, 0.3)
            joint4 *= (1 + reach_factor * 0.3)  # 최대 30% 증폭
        
        # 5. Joint4 제한
        joint4 = np.clip(joint4, -1.0, 1.0)
        
        # 통계 업데이트
        self.stats['orientation_direct_mapping'] += 1
        
        return [joint1, joint2, joint3, joint4]
    
    def apply_joint4_filter(self, target_joints):
        """Joint4 전용 필터 적용"""
        self.filter_history['joint_targets'].append(target_joints)
        self.filter_history['joint4_targets'].append(target_joints[3])
        
        if len(self.filter_history['joint_targets']) < 8:
            return target_joints
        
        recent_targets = np.array(list(self.filter_history['joint_targets'])[-15:])
        recent_j4 = np.array(list(self.filter_history['joint4_targets'])[-10:])
        
        filtered_joints = []
        
        # Joint 1-3: 강한 필터링
        for joint_idx in range(3):
            joint_history = recent_targets[:, joint_idx]
            median_value = np.median(joint_history[-5:])
            
            if len(joint_history) >= 10:
                try:
                    filtered_value = filtfilt(self.filter_b, self.filter_a, joint_history)[-1]
                    final_value = 0.3 * median_value + 0.7 * filtered_value
                    filtered_joints.append(final_value)
                except:
                    filtered_joints.append(median_value)
            else:
                filtered_joints.append(median_value)
        
        # Joint4: 약한 필터링 (더 반응적)
        if len(recent_j4) >= 5:
            try:
                # Joint4 전용 필터 사용
                filtered_j4 = filtfilt(self.j4_filter_b, self.j4_filter_a, recent_j4)[-1]
                # 중간값과 필터값의 가중치 조정 (필터값 우선)
                median_j4 = np.median(recent_j4[-3:])
                final_j4 = 0.1 * median_j4 + 0.9 * filtered_j4
                filtered_joints.append(final_j4)
            except:
                filtered_joints.append(target_joints[3])
        else:
            filtered_joints.append(target_joints[3])
        
        return filtered_joints
    
    def orientation_enhanced_control_loop(self):
        """Orientation 강화 제어 루프"""
        rate = rospy.Rate(120)
        
        while not rospy.is_shutdown():
            loop_start_time = time.time()
            
            if self.vr_data['hand_pose']['calibrated']:
                vr_pos_delta, vr_ori_delta = self.get_vr_deltas()
                
                if vr_pos_delta is not None and vr_ori_delta is not None:
                    try:
                        # 직관적 Joint4 제어
                        raw_target_joints = self.intuitive_joint4_control(vr_pos_delta, vr_ori_delta)
                        
                        # Joint4 특화 필터링
                        filtered_target_joints = self.apply_joint4_filter(raw_target_joints)
                        
                        # 부드러운 업데이트 (Joint4는 더 빠르게)
                        for i in range(4):
                            joint_error = filtered_target_joints[i] - self.robot_joints[i]
                            
                            # Joint4는 더 빠른 반응
                            if i == 3:
                                max_change = self.joint4_mapping['max_change_rate']
                                smooth_factor = self.joint4_mapping['smoothing_factor']
                            else:
                                max_change = self.safety_params['max_joint_speed']
                                smooth_factor = self.safety_params['smooth_factor']
                            
                            joint_error = np.clip(joint_error, -max_change, max_change)
                            self.robot_joints[i] += joint_error * smooth_factor
                            
                            # 안전 체크
                            if i == 3:  # Joint4
                                self.robot_joints[i] = np.clip(self.robot_joints[i], -1.2, 1.2)
                            else:
                                if abs(self.robot_joints[i]) > 1.3:
                                    self.robot_joints[i] = np.sign(self.robot_joints[i]) * 1.3
                        
                    except Exception as e:
                        rospy.logwarn(f"제어 오류: {e}")
            
            self.update_gripper_control()
            self.send_to_mujoco()
            
            loop_time = time.time() - loop_start_time
            self.stats['control_frequency'] = 1.0 / max(loop_time, 0.001)
            
            rate.sleep()
    
    def get_vr_deltas(self):
        """VR 델타 계산"""
        if not self.vr_data['hand_pose']['calibrated']:
            return None, None
        
        current_pos = self.vr_data['hand_pose']['position']
        initial_pos = self.vr_data['hand_pose']['initial_position']
        position_delta = (current_pos - initial_pos) * self.safety_params['position_scale']
        position_delta = np.clip(position_delta, -0.3, 0.3)
        
        current_ori = self.vr_data['hand_pose']['orientation']
        initial_ori = self.vr_data['hand_pose']['initial_orientation']
        
        current_euler = tf_trans.euler_from_quaternion(current_ori)
        initial_euler = tf_trans.euler_from_quaternion(initial_ori)
        orientation_delta = np.array(current_euler) - np.array(initial_euler)
        
        # Orientation 델타는 제한하지 않음 (Joint4 반응성을 위해)
        orientation_delta = np.clip(orientation_delta, -2.0, 2.0)
        
        return position_delta, orientation_delta
    
    # 나머지 메서드들은 V3.1과 동일
    def setup_socket_server(self):
        """소켓 서버 설정"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 12345))
            self.server_socket.listen(5)
            self.clients = []
            
            accept_thread = threading.Thread(target=self.accept_clients, daemon=True)
            accept_thread.start()
            
            print("✅ 소켓 서버 시작: 포트 12345")
        except Exception as e:
            print(f"❌ 소켓 서버 오류: {e}")
    
    def accept_clients(self):
        """클라이언트 수락"""
        while True:
            try:
                client, addr = self.server_socket.accept()
                self.clients.append(client)
                print(f"🔗 MuJoCo 클라이언트 연결: {addr}")
            except:
                break
    
    def setup_ros_topics(self):
        """ROS 토픽 설정"""
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, self.hand_pose_callback)
        
        try:
            from quest2ros.msg import OVR2ROSInputs
            rospy.Subscriber('/q2r_left_hand_inputs', OVR2ROSInputs, self.input_callback)
            print("✅ VR 입력 토픽 구독됨")
        except ImportError:
            print("⚠️ OVR2ROSInputs 메시지 없음")
        
        print("✅ ROS 토픽 설정 완료")
    
    def hand_pose_callback(self, msg):
        """VR 손 Pose 콜백 (Orientation 히스토리 추가)"""
        current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        current_orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        pose_data = {
            'position': current_position,
            'orientation': current_orientation,
            'timestamp': time.time()
        }
        self.vr_data['hand_pose']['pose_history'].append(pose_data)
        self.vr_data['hand_pose']['orientation_history'].append(current_orientation)
        
        if len(self.vr_data['hand_pose']['pose_history']) >= 8:
            recent_poses = list(self.vr_data['hand_pose']['pose_history'])[-12:]
            
            current_time = time.time()
            weights = []
            for pose in recent_poses:
                time_diff = current_time - pose['timestamp']
                weight = np.exp(-time_diff * 2.0)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            positions = [p['position'] for p in recent_poses]
            orientations = [p['orientation'] for p in recent_poses]
            
            smoothed_position = np.average(positions, axis=0, weights=weights)
            
            # Orientation은 덜 스무딩 (Joint4 반응성)
            recent_ori_weights = weights[-6:]  # 최근 6개만
            recent_ori_weights = recent_ori_weights / np.sum(recent_ori_weights)
            smoothed_orientation = np.average(orientations[-6:], axis=0, weights=recent_ori_weights)
            smoothed_orientation = smoothed_orientation / np.linalg.norm(smoothed_orientation)
        else:
            smoothed_position = current_position
            smoothed_orientation = current_orientation
        
        self.vr_data['hand_pose']['position'] = smoothed_position
        self.vr_data['hand_pose']['orientation'] = smoothed_orientation
        
        if not self.vr_data['hand_pose']['calibrated']:
            self.vr_data['hand_pose']['initial_position'] = smoothed_position.copy()
            self.vr_data['hand_pose']['initial_orientation'] = smoothed_orientation.copy()
            self.vr_data['hand_pose']['calibrated'] = True
            print(f"🖐 VR 컨트롤러 캘리브레이션 완료")
    
    def input_callback(self, msg):
        """VR 입력 콜백"""
        try:
            if hasattr(msg, 'trigger'):
                self.vr_data['inputs']['trigger'] = msg.trigger
            if hasattr(msg, 'button_upper'):
                self.vr_data['inputs']['button_upper'] = msg.button_upper
            if hasattr(msg, 'button_lower'):
                self.vr_data['inputs']['button_lower'] = msg.button_lower
            
            if (self.vr_data['inputs']['button_upper'] and 
                self.vr_data['inputs']['button_lower']):
                self.recalibrate()
                
        except Exception as e:
            rospy.logwarn(f"입력 처리 오류: {e}")
    
    def recalibrate(self):
        """재캘리브레이션"""
        if self.vr_data['hand_pose']['position'] is not None:
            self.vr_data['hand_pose']['initial_position'] = self.vr_data['hand_pose']['position'].copy()
            self.vr_data['hand_pose']['initial_orientation'] = self.vr_data['hand_pose']['orientation'].copy()
            self.vr_data['hand_pose']['pose_history'].clear()
            self.vr_data['hand_pose']['orientation_history'].clear()
            print("🔄 VR 컨트롤러 재캘리브레이션 완료!")
    
    def update_gripper_control(self):
        """그리퍼 제어"""
        trigger_value = self.vr_data['inputs']['trigger']
        self.gripper_value = -0.01 + (trigger_value * 0.029)
        
        if self.vr_data['inputs']['button_upper']:
            self.gripper_value = 0.019
    
    def send_to_mujoco(self):
        """MuJoCo로 데이터 전송"""
        if self.clients:
            data = {
                'joint_angles': self.robot_joints,
                'gripper': self.gripper_value,
                'vr_status': {
                    'calibrated': self.vr_data['hand_pose']['calibrated'],
                    'trigger_value': self.vr_data['inputs']['trigger'],
                    'button_upper': self.vr_data['inputs']['button_upper'],
                    'button_lower': self.vr_data['inputs']['button_lower']
                },
                'performance': self.stats,
                'timestamp': rospy.Time.now().to_sec()
            }
            
            json_data = json.dumps(data) + '\n'
            
            failed_clients = []
            for client in self.clients:
                try:
                    client.sendall(json_data.encode())
                except:
                    failed_clients.append(client)
            
            for client in failed_clients:
                if client in self.clients:
                    self.clients.remove(client)
    
    def debug_loop(self):
        """디버그 출력"""
        while not rospy.is_shutdown():
            time.sleep(4.0)
            
            print(f"\n🤏 === Joint4 직관적 제어 V3.2 상태 ===")
            print(f"🖐 VR 캘리브레이션: {'✅' if self.vr_data['hand_pose']['calibrated'] else '❌'}")
            print(f"🎮 트리거: {self.vr_data['inputs']['trigger']:.2f}")
            print(f"🤖 조인트: J1={self.robot_joints[0]:.3f}, J2={self.robot_joints[1]:.3f}, J3={self.robot_joints[2]:.3f}, J4={self.robot_joints[3]:.3f}")
            print(f"🎯 Joint4 Pitch 민감도: {self.joint4_mapping['pitch_sensitivity']:.1f}")
            print(f"📊 Orientation 직접 매핑: {self.stats['orientation_direct_mapping']}회")
            print(f"⚡ 제어 주파수: {self.stats['control_frequency']:.1f}Hz")
            
            if self.vr_data['hand_pose']['calibrated']:
                vr_pos_delta, vr_ori_delta = self.get_vr_deltas()
                if vr_pos_delta is not None and vr_ori_delta is not None:
                    print(f"📍 VR 위치: X={vr_pos_delta[0]:+.3f}, Y={vr_pos_delta[1]:+.3f}, Z={vr_pos_delta[2]:+.3f}")
                    print(f"🔄 VR 회전: Roll={vr_ori_delta[0]:+.3f}, Pitch={vr_ori_delta[1]:+.3f}, Yaw={vr_ori_delta[2]:+.3f}")
                    print(f"🤏 Joint4 목표: {self.robot_joints[3]:+.3f} (Pitch {vr_ori_delta[1]:+.3f} 기반)")

if __name__ == "__main__":
    bridge = IntuitiveJoint4VRBridge()
    
    print("\n🤏 === Joint4 직관적 제어 강화 시스템 V3.2 ===")
    print("🎯 손목 회전 → Joint4 정밀 매핑")
    print("⬆️ 손목 위로: Joint4 음수")
    print("⬇️ 손목 아래로: Joint4 양수")
    print("↔️ Roll/Yaw: Joint4 보조 제어")
    print("🖐 왼쪽 VR 컨트롤러 → OpenManipulator-X")
    print("🎯 트리거 → 그리퍼 제어")
    print("🔄 A+B 버튼 → 재캘리브레이션")
    
    try:
        while not rospy.is_shutdown():
            try:
                key = input().strip().lower()
                
                if key == 'c':
                    bridge.recalibrate()
                elif key == 'r':
                    bridge.robot_joints = [0.0, 0.0, 0.0, 0.0]
                    bridge.joint_velocities = [0.0, 0.0, 0.0, 0.0]
                    bridge.gripper_value = -0.01
                    print("🔄 로봇 리셋됨")
                elif key == 'q':
                    break
                    
            except (EOFError, KeyboardInterrupt):
                break
                
    except:
        pass
    
    print("🏁 Joint4 직관적 제어 시스템 종료")
