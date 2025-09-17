#!/usr/bin/env python3
"""
🎯 Dual Arm VR Bridge - X축 움직임 개선 버전
- 새 데이터셋 기반 X축 매핑 (상관계수 0.94)
- X축 앞뒤 움직임으로 Joint2,3 정확한 제어
- 왼쪽/오른쪽 컨트롤러 동시 처리
"""

import rospy
import numpy as np
import json
import socket
import threading
import time
from collections import deque
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt

class DualArmIntuitiveVRBridge:
    def __init__(self):
        rospy.init_node('dual_arm_intuitive_vr_bridge')
        
        print("🎯 Dual Arm Intuitive VR Bridge 시작")
        print("🤏 양팔 독립 제어 + Joint4 정밀 매핑")
        
        # 소켓 서버 설정 (하나의 소켓으로 통합)
        self.setup_socket_server()
        
        # 왼팔 VR 데이터
        self.left_vr_data = self.create_arm_data_structure()
        # 오른팔 VR 데이터  
        self.right_vr_data = self.create_arm_data_structure()
        
        # 왼팔 로봇 조인트
        self.left_robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.left_gripper_value = -0.01
        
        # 오른팔 로봇 조인트
        self.right_robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.right_gripper_value = -0.01
        
        # 안전 파라미터 (test3.py와 동일)
        self.safety_params = {
            'max_joint_speed': 0.08,
            'position_scale': np.array([0.7, 0.7, 0.7]),
            'smooth_factor': 0.08,
            'z_axis_gain': 0.4,
            'joint1_gain': 0.55,
            'safety_margin': 0.9
        }
        
        # Z축 매핑 (test3.py와 동일)
        self.z_axis_mapping = {
            'down_threshold': -0.08,
            'up_threshold': 0.08,
            'joint2_down_gain': 0.35,
            'joint3_down_gain': 0.25,
            'joint2_up_gain': -0.25,
            'joint3_up_gain': -0.35,
        }
        
        # Joint4 직관적 매핑 (test3.py와 동일)
        self.joint4_mapping = {
            'pitch_sensitivity': 1.2,
            'roll_sensitivity': 0.3,
            'yaw_sensitivity': 0.2,
            'direct_influence': 0.7,
            'smoothing_factor': 0.15,
            'deadzone': 0.05,
            'max_change_rate': 0.12
        }
        
        # Orientation 패턴 (test3.py와 동일)
        self.orientation_patterns = {
            'pitch_down': {'range': [0.5, 1.5], 'joint4_target': 0.8},
            'pitch_up': {'range': [-1.5, -0.5], 'joint4_target': -0.8},
            'neutral': {'range': [-0.3, 0.3], 'joint4_target': 0.0},
            'roll_left': {'range': [0.3, 1.0], 'joint4_modifier': 0.2},
            'roll_right': {'range': [-1.0, -0.3], 'joint4_modifier': -0.2}
        }
        
        # 매핑 데이터 로드 및 KD-Tree 구성
        self.load_orientation_enhanced_data()
        self.build_orientation_trees()
        
        # 필터 설정
        self.setup_filters()
        
        # 성능 통계
        self.stats = {
            'left': {
                'orientation_direct_mapping': 0,
                'position_mapping': 0,
            },
            'right': {
                'orientation_direct_mapping': 0,
                'position_mapping': 0,
            },
            'control_frequency': 0.0
        }
        
        # ROS 토픽 설정 (양팔)
        self.setup_ros_topics()
        
        # 제어 루프 시작
        self.control_thread = threading.Thread(target=self.dual_arm_control_loop, daemon=True)
        self.control_thread.start()
        
        # 디버그 스레드
        self.debug_thread = threading.Thread(target=self.debug_loop, daemon=True)
        self.debug_thread.start()
        
        print("✅ Dual Arm Intuitive VR Bridge 준비 완료")
    
    def create_arm_data_structure(self):
        """각 팔용 데이터 구조 생성"""
        return {
            'hand_pose': {
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'initial_position': None,
                'initial_orientation': None,
                'calibrated': False,
                'pose_history': deque(maxlen=15),
                'orientation_history': deque(maxlen=20)
            },
            'inputs': {
                'trigger': 0.0,
                'button_upper': False,
                'button_lower': False
            }
        }
    
    def setup_filters(self):
        """필터 설정 (양팔용)"""
        self.filter_freq = 5.0
        self.filter_order = 3
        self.joint4_filter_freq = 10.0
        self.joint4_filter_order = 2
        
        nyquist = 120.0 / 2
        
        normalized_freq = self.filter_freq / nyquist
        self.filter_b, self.filter_a = butter(self.filter_order, normalized_freq, btype='low')
        
        j4_normalized_freq = self.joint4_filter_freq / nyquist
        self.j4_filter_b, self.j4_filter_a = butter(self.joint4_filter_order, j4_normalized_freq, btype='low')
        
        # 왼팔 필터 히스토리
        self.left_filter_history = {
            'joint_targets': deque(maxlen=30),
            'joint4_targets': deque(maxlen=15),
            'vr_deltas': deque(maxlen=15)
        }
        
        # 오른팔 필터 히스토리
        self.right_filter_history = {
            'joint_targets': deque(maxlen=30),
            'joint4_targets': deque(maxlen=15),
            'vr_deltas': deque(maxlen=15)
        }
    
    def load_orientation_enhanced_data(self):
        """Orientation 강화 매핑 데이터 (test3.py와 동일)"""
        print("📁 Orientation 강화 매핑 데이터 로드 중...")
        
        base_data = [
            {"vr_pos": [0.001, -0.013, -0.0003], "vr_ori": [-0.022, 0.043, -0.054], "joints": [0.0, 0.0, 0.0, 0.0]},
            {"vr_pos": [0.019, -0.003, -0.021], "vr_ori": [0.149, 0.689, 0.109], "joints": [0.0, 0.0, 0.0, 0.5]},
            {"vr_pos": [-0.036, -0.012, 0.048], "vr_ori": [-0.037, -0.672, -0.056], "joints": [0.0, 0.0, 0.0, -0.5]},
        ]
        
        orientation_samples = []
        
        # Pitch 변화 샘플
        for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]:
            for pitch in np.linspace(-1.2, 1.2, 9):
                joint4 = pitch * 0.7
                sample = {
                    "vr_pos": pos,
                    "vr_ori": [0.0, pitch, 0.0],
                    "joints": [0.0, 0.0, 0.0, joint4]
                }
                orientation_samples.append(sample)
        
        # Roll 변화 샘플
        for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]:
            for roll in [-0.8, -0.4, 0.0, 0.4, 0.8]:
                joint4 = roll * 0.3
                sample = {
                    "vr_pos": pos,
                    "vr_ori": [roll, 0.0, 0.0],
                    "joints": [0.0, 0.0, 0.0, joint4]
                }
                orientation_samples.append(sample)
        
        # 복합 orientation 샘플
        complex_orientations = [
            {"ori": [0.3, 0.8, 0.1], "j4": 0.7},
            {"ori": [-0.3, -0.8, -0.1], "j4": -0.7},
            {"ori": [0.5, 0.0, 0.5], "j4": 0.3},
            {"ori": [-0.5, 0.0, -0.5], "j4": -0.3},
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
        
        # 위치별 orientation 변화
        position_orientation_pairs = [
            {"pos": [0.15, 0.0, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [0.0, 0.3, -0.3, 0.6]},
            {"pos": [0.15, 0.0, 0.0], "ori": [0.0, -0.8, 0.0], "joints": [0.0, 0.3, -0.3, -0.6]},
            {"pos": [0.0, 0.15, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [0.4, 0.0, 0.0, 0.6]},
            {"pos": [0.0, -0.15, 0.0], "ori": [0.0, 0.8, 0.0], "joints": [-0.4, 0.0, 0.0, 0.6]},
            {"pos": [0.0, 0.0, -0.15], "ori": [0.0, 0.8, 0.0], "joints": [0.0, 0.3, 0.2, 0.6]},
            {"pos": [0.0, 0.0, -0.15], "ori": [0.0, -0.8, 0.0], "joints": [0.0, 0.3, 0.2, -0.6]},
        ]
        
        for pair in position_orientation_pairs:
            orientation_samples.append({
                "vr_pos": pair["pos"],
                "vr_ori": pair["ori"],
                "joints": pair["joints"]
            })
        
        self.mapping_data = base_data + orientation_samples
        print(f"✅ {len(self.mapping_data)}개 매핑 데이터 로드 완료")
    
    def build_orientation_trees(self):
        """KD-Tree 구성 (test3.py와 동일)"""
        print("🔍 Orientation 특화 KD-Tree 구성 중...")
        
        positions = np.array([sample['vr_pos'] for sample in self.mapping_data])
        self.position_tree = cKDTree(positions)
        
        orientations = np.array([sample['vr_ori'] for sample in self.mapping_data])
        self.orientation_tree = cKDTree(orientations)
        
        pitch_features = []
        for sample in self.mapping_data:
            pitch_weighted = [
                sample['vr_ori'][0] * 0.3,
                sample['vr_ori'][1] * 1.5,
                sample['vr_ori'][2] * 0.2
            ]
            pitch_features.append(pitch_weighted)
        self.pitch_tree = cKDTree(np.array(pitch_features))
        
        combined_features = []
        for sample in self.mapping_data:
            combined = (
                list(np.array(sample['vr_pos']) * 0.5) +
                list(np.array(sample['vr_ori']) * 1.2)
            )
            combined_features.append(combined)
        self.combined_tree = cKDTree(np.array(combined_features))
        
        print("🔍 KD-Tree 구성 완료")
    
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
        """ROS 토픽 설정 - 양팔"""
        # 왼팔 토픽
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, 
                        lambda msg: self.hand_pose_callback(msg, 'left'))
        
        # 오른팔 토픽
        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, 
                        lambda msg: self.hand_pose_callback(msg, 'right'))
        
        try:
            from quest2ros.msg import OVR2ROSInputs
            # 왼팔 입력
            rospy.Subscriber('/q2r_left_hand_inputs', OVR2ROSInputs, 
                           lambda msg: self.input_callback(msg, 'left'))
            # 오른팔 입력
            rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, 
                           lambda msg: self.input_callback(msg, 'right'))
            print("✅ 양팔 VR 입력 토픽 구독됨")
        except ImportError:
            print("⚠️ OVR2ROSInputs 메시지 없음")
        
        print("✅ 양팔 ROS 토픽 설정 완료")
    
    def hand_pose_callback(self, msg, arm_side):
        """VR 손 Pose 콜백 (test3.py 로직 적용)"""
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
        
        # 팔 선택
        vr_data = self.left_vr_data if arm_side == 'left' else self.right_vr_data
        
        pose_data = {
            'position': current_position,
            'orientation': current_orientation,
            'timestamp': time.time()
        }
        vr_data['hand_pose']['pose_history'].append(pose_data)
        vr_data['hand_pose']['orientation_history'].append(current_orientation)
        
        # 스무딩 (test3.py와 동일)
        if len(vr_data['hand_pose']['pose_history']) >= 8:
            recent_poses = list(vr_data['hand_pose']['pose_history'])[-12:]
            
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
            recent_ori_weights = weights[-6:]
            recent_ori_weights = recent_ori_weights / np.sum(recent_ori_weights)
            smoothed_orientation = np.average(orientations[-6:], axis=0, weights=recent_ori_weights)
            smoothed_orientation = smoothed_orientation / np.linalg.norm(smoothed_orientation)
        else:
            smoothed_position = current_position
            smoothed_orientation = current_orientation
        
        vr_data['hand_pose']['position'] = smoothed_position
        vr_data['hand_pose']['orientation'] = smoothed_orientation
        
        # 초기 캘리브레이션
        if not vr_data['hand_pose']['calibrated']:
            vr_data['hand_pose']['initial_position'] = smoothed_position.copy()
            vr_data['hand_pose']['initial_orientation'] = smoothed_orientation.copy()
            vr_data['hand_pose']['calibrated'] = True
            print(f"🖐 {arm_side.upper()} VR 컨트롤러 캘리브레이션 완료")
    
    def input_callback(self, msg, arm_side):
        """VR 입력 콜백"""
        vr_data = self.left_vr_data if arm_side == 'left' else self.right_vr_data
        
        try:
            if hasattr(msg, 'trigger'):
                vr_data['inputs']['trigger'] = msg.trigger
            if hasattr(msg, 'button_upper'):
                vr_data['inputs']['button_upper'] = msg.button_upper
            if hasattr(msg, 'button_lower'):
                vr_data['inputs']['button_lower'] = msg.button_lower
            
            # A+B 버튼으로 재캘리브레이션
            if (vr_data['inputs']['button_upper'] and 
                vr_data['inputs']['button_lower']):
                self.recalibrate(arm_side)
                
        except Exception as e:
            rospy.logwarn(f"{arm_side} 입력 처리 오류: {e}")
    
    def recalibrate(self, arm_side):
        """재캘리브레이션"""
        vr_data = self.left_vr_data if arm_side == 'left' else self.right_vr_data
        
        if vr_data['hand_pose']['position'] is not None:
            vr_data['hand_pose']['initial_position'] = vr_data['hand_pose']['position'].copy()
            vr_data['hand_pose']['initial_orientation'] = vr_data['hand_pose']['orientation'].copy()
            vr_data['hand_pose']['pose_history'].clear()
            vr_data['hand_pose']['orientation_history'].clear()
            print(f"🔄 {arm_side.upper()} VR 재캘리브레이션 완료!")
    
    def get_vr_deltas(self, vr_data):
        """VR 델타 계산 (test3.py와 동일)"""
        if not vr_data['hand_pose']['calibrated']:
            return None, None
        
        current_pos = vr_data['hand_pose']['position']
        initial_pos = vr_data['hand_pose']['initial_position']
        position_delta = (current_pos - initial_pos) * self.safety_params['position_scale']
        position_delta = np.clip(position_delta, -0.3, 0.3)
        
        current_ori = vr_data['hand_pose']['orientation']
        initial_ori = vr_data['hand_pose']['initial_orientation']
        
        current_euler = tf_trans.euler_from_quaternion(current_ori)
        initial_euler = tf_trans.euler_from_quaternion(initial_ori)
        orientation_delta = np.array(current_euler) - np.array(initial_euler)
        orientation_delta = np.clip(orientation_delta, -2.0, 2.0)
        
        return position_delta, orientation_delta
    
    def intuitive_joint4_control(self, vr_pos_delta, vr_ori_delta, arm_side):
        """Joint4 직관적 제어 + 새 데이터 기반 X축 매핑"""
        # Joint 1 계산 (기존과 동일)
        joint1 = np.arctan2(vr_pos_delta[1], vr_pos_delta[0] + 0.25) * self.safety_params['joint1_gain']
        joint1 = np.clip(joint1, -0.65, 0.65)
        
        # X-Z 평면 기반 계산
        x_delta = vr_pos_delta[0]
        z_delta = vr_pos_delta[2]
        
        # === 새 데이터 기반 X축 매핑 (상관계수 0.94) ===
        # X와 J2는 강한 양의 상관관계
        # X와 J3는 중간 음의 상관관계 + Z 영향
        
        if x_delta > 0.01:  # 앞으로 (X 양수)
            # 데이터: X=0.232 → J2=0.521, J3=-0.152
            # 기울기: J2/X ≈ 2.2, J3/X ≈ -0.65
            x_normalized = min(x_delta / 0.25, 1.0)
            joint2_x = x_normalized * 0.55  # 2.2 * 0.25 = 0.55
            joint3_x = x_normalized * -0.16  # -0.65 * 0.25 = -0.16
            
        elif x_delta < -0.01:  # 뒤로 (X 음수)
            # 데이터: X=-0.178 → J2=-0.817, J3=0.793
            # 기울기: J2/X ≈ 4.6, J3/X ≈ -4.5
            x_normalized = min(abs(x_delta) / 0.20, 1.0)
            joint2_x = -x_normalized * 0.92  # -4.6 * 0.20 = -0.92
            joint3_x = x_normalized * 0.90   # 4.5 * 0.20 = 0.90
            
        else:  # 중립 범위
            joint2_x = 0.0
            joint3_x = 0.0
        
        # === Z축 매핑 (상관계수 J3와 -0.80) ===
        # Z는 주로 J3에 영향, J2는 약간
        if z_delta > 0.08:  # 위로
            z_normalized = min(z_delta / 0.3, 1.0)
            joint2_z = z_normalized * 0.15  # 약한 영향
            joint3_z = z_normalized * -0.65  # 강한 음의 영향
            
        elif z_delta < -0.08:  # 아래로
            z_normalized = min(abs(z_delta) / 0.3, 1.0)
            joint2_z = z_normalized * 0.20
            joint3_z = z_normalized * 0.55
            
        else:  # 중립
            joint2_z = 0.0
            joint3_z = 0.0
        
        # === X-Z 복합 효과 ===
        # X가 주도적, Z는 보조적 역할
        if abs(x_delta) > 0.01 and abs(z_delta) > 0.05:
            # 복합 동작 시 조정
            combo_factor = 0.85  # 복합 동작 시 약간 감소
            joint2 = (joint2_x + joint2_z * 0.5) * combo_factor
            joint3 = (joint3_x + joint3_z * 0.8) * combo_factor
        else:
            # 단일 축 움직임
            joint2 = joint2_x + joint2_z
            joint3 = joint3_x + joint3_z
        
        # === 작은 움직임 보간 (KD-Tree) ===
        if abs(x_delta) <= 0.01 and abs(z_delta) <= 0.08:
            distances, indices = self.position_tree.query(vr_pos_delta, k=min(4, len(self.mapping_data)))
            if isinstance(distances, float):
                distances = [distances]
                indices = [indices]
            weights = 1.0 / (np.array(distances) + 1e-6)
            weights = weights / np.sum(weights)
            joint2_tree = sum(weights[i] * self.mapping_data[idx]['joints'][1] for i, idx in enumerate(indices))
            joint3_tree = sum(weights[i] * self.mapping_data[idx]['joints'][2] for i, idx in enumerate(indices))
            # 보간 값과 혼합
            joint2 = joint2 * 0.6 + joint2_tree * 0.4
            joint3 = joint3 * 0.6 + joint3_tree * 0.4
        
        # Joint2, Joint3 최종 클리핑
        joint2 = np.clip(joint2, -0.85, 0.55)  # 실제 데이터 범위 기준
        joint3 = np.clip(joint3, -0.85, 0.97)
        
        # 디버그 출력
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 60 == 0:  # 0.5초마다
            side_str = "RIGHT" if arm_side == 'right' else "LEFT "
            print(f"[{side_str}] X={x_delta:.3f}, Z={z_delta:.3f}, Reach={reach_distance:.3f}")
            print(f"        → J1={joint1:.3f}, J2={joint2:.3f}, J3={joint3:.3f}")
        
        # Joint4: Orientation 기반 직관적 제어 (기존과 동일)
        pitch = vr_ori_delta[1]
        roll = vr_ori_delta[0]
        yaw = vr_ori_delta[2]
        
        # 데드존 적용
        if abs(pitch) < self.joint4_mapping['deadzone']:
            pitch = 0.0
        
        # Pitch → Joint4 직접 매핑
        joint4_from_pitch = pitch * self.joint4_mapping['pitch_sensitivity']
        joint4_from_roll = roll * self.joint4_mapping['roll_sensitivity']
        joint4_from_yaw = yaw * self.joint4_mapping['yaw_sensitivity']
        
        # Orientation KD-Tree 보간
        pitch_weighted_query = [
            roll * 0.3,
            pitch * 1.5,
            yaw * 0.2
        ]
        
        ori_distances, ori_indices = self.pitch_tree.query(pitch_weighted_query, k=min(6, len(self.mapping_data)))
        
        if isinstance(ori_distances, float):
            ori_distances = [ori_distances]
            ori_indices = [ori_indices]
        
        ori_weights = 1.0 / (np.array(ori_distances) + 1e-6)
        ori_weights = ori_weights / np.sum(ori_weights)
        
        joint4_from_tree = sum(ori_weights[i] * self.mapping_data[idx]['joints'][3] 
                               for i, idx in enumerate(ori_indices))
        
        # 최종 Joint4 결합
        joint4_direct = joint4_from_pitch + joint4_from_roll * 0.3 + joint4_from_yaw * 0.2
        
        joint4 = (
            self.joint4_mapping['direct_influence'] * joint4_direct +
            (1 - self.joint4_mapping['direct_influence']) * joint4_from_tree
        )
        
        # 위치 기반 보정
        reach = np.sqrt(vr_pos_delta[0]**2 + vr_pos_delta[1]**2)
        if reach > 0.15:
            reach_factor = min((reach - 0.15) / 0.15, 0.3)
            joint4 *= (1 + reach_factor * 0.3)
        
        joint4 = np.clip(joint4, -1.0, 1.0)
        
        # 통계 업데이트
        self.stats[arm_side]['orientation_direct_mapping'] += 1
        
        return [joint1, joint2, joint3, joint4]
    
    def apply_joint4_filter(self, target_joints, filter_history):
        """Joint4 전용 필터 적용"""
        filter_history['joint_targets'].append(target_joints)
        filter_history['joint4_targets'].append(target_joints[3])
        
        if len(filter_history['joint_targets']) < 8:
            return target_joints
        
        recent_targets = np.array(list(filter_history['joint_targets'])[-15:])
        recent_j4 = np.array(list(filter_history['joint4_targets'])[-10:])
        
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
        
        # Joint4: 약한 필터링
        if len(recent_j4) >= 5:
            try:
                filtered_j4 = filtfilt(self.j4_filter_b, self.j4_filter_a, recent_j4)[-1]
                median_j4 = np.median(recent_j4[-3:])
                final_j4 = 0.1 * median_j4 + 0.9 * filtered_j4
                filtered_joints.append(final_j4)
            except:
                filtered_joints.append(target_joints[3])
        else:
            filtered_joints.append(target_joints[3])
        
        return filtered_joints
    
    def update_gripper_control(self, vr_data):
        """그리퍼 제어"""
        trigger_value = vr_data['inputs']['trigger']
        gripper_value = -0.01 + (trigger_value * 0.029)
        
        if vr_data['inputs']['button_upper']:
            gripper_value = 0.019
        
        return gripper_value
    
    def dual_arm_control_loop(self):
        """양팔 제어 루프"""
        rate = rospy.Rate(120)
        
        while not rospy.is_shutdown():
            loop_start_time = time.time()
            
            # 왼팔 제어
            if self.left_vr_data['hand_pose']['calibrated']:
                left_pos_delta, left_ori_delta = self.get_vr_deltas(self.left_vr_data)
                
                if left_pos_delta is not None and left_ori_delta is not None:
                    try:
                        # Joint4 직관적 제어
                        raw_left_joints = self.intuitive_joint4_control(
                            left_pos_delta, left_ori_delta, 'left'
                        )
                        
                        # 필터링
                        filtered_left_joints = self.apply_joint4_filter(
                            raw_left_joints, self.left_filter_history
                        )
                        
                        # 부드러운 업데이트
                        for i in range(4):
                            joint_error = filtered_left_joints[i] - self.left_robot_joints[i]
                            
                            if i == 3:  # Joint4
                                max_change = self.joint4_mapping['max_change_rate']
                                smooth_factor = self.joint4_mapping['smoothing_factor']
                            else:
                                max_change = self.safety_params['max_joint_speed']
                                smooth_factor = self.safety_params['smooth_factor']
                            
                            joint_error = np.clip(joint_error, -max_change, max_change)
                            self.left_robot_joints[i] += joint_error * smooth_factor
                            
                            # 안전 체크
                            if i == 3:
                                self.left_robot_joints[i] = np.clip(self.left_robot_joints[i], -1.2, 1.2)
                            else:
                                if abs(self.left_robot_joints[i]) > 1.3:
                                    self.left_robot_joints[i] = np.sign(self.left_robot_joints[i]) * 1.3
                        
                    except Exception as e:
                        rospy.logwarn(f"왼팔 제어 오류: {e}")
            
            # 오른팔 제어 (왼팔과 동일한 로직)
            if self.right_vr_data['hand_pose']['calibrated']:
                right_pos_delta, right_ori_delta = self.get_vr_deltas(self.right_vr_data)
                
                if right_pos_delta is not None and right_ori_delta is not None:
                    try:
                        # Joint4 직관적 제어
                        raw_right_joints = self.intuitive_joint4_control(
                            right_pos_delta, right_ori_delta, 'right'
                        )
                        
                        # 필터링
                        filtered_right_joints = self.apply_joint4_filter(
                            raw_right_joints, self.right_filter_history
                        )
                        
                        # 부드러운 업데이트
                        for i in range(4):
                            joint_error = filtered_right_joints[i] - self.right_robot_joints[i]
                            
                            if i == 3:  # Joint4
                                max_change = self.joint4_mapping['max_change_rate']
                                smooth_factor = self.joint4_mapping['smoothing_factor']
                            else:
                                max_change = self.safety_params['max_joint_speed']
                                smooth_factor = self.safety_params['smooth_factor']
                            
                            joint_error = np.clip(joint_error, -max_change, max_change)
                            self.right_robot_joints[i] += joint_error * smooth_factor
                            
                            # 안전 체크
                            if i == 3:
                                self.right_robot_joints[i] = np.clip(self.right_robot_joints[i], -1.2, 1.2)
                            else:
                                if abs(self.right_robot_joints[i]) > 1.3:
                                    self.right_robot_joints[i] = np.sign(self.right_robot_joints[i]) * 1.3
                        
                    except Exception as e:
                        rospy.logwarn(f"오른팔 제어 오류: {e}")
            
            # 그리퍼 업데이트
            self.left_gripper_value = self.update_gripper_control(self.left_vr_data)
            self.right_gripper_value = self.update_gripper_control(self.right_vr_data)
            
            # MuJoCo로 전송
            self.send_to_mujoco()
            
            loop_time = time.time() - loop_start_time
            self.stats['control_frequency'] = 1.0 / max(loop_time, 0.001)
            
            rate.sleep()
    
    def send_to_mujoco(self):
        """MuJoCo로 양팔 데이터 전송 (단일 소켓)"""
        if self.clients:
            # 양팔 데이터를 하나의 패킷으로 전송
            data = {
                'left_arm': {
                    'joint_angles': self.left_robot_joints,
                    'gripper': self.left_gripper_value,
                    'calibrated': self.left_vr_data['hand_pose']['calibrated'],
                    'trigger_value': self.left_vr_data['inputs']['trigger'],
                },
                'right_arm': {
                    'joint_angles': self.right_robot_joints,
                    'gripper': self.right_gripper_value,
                    'calibrated': self.right_vr_data['hand_pose']['calibrated'],
                    'trigger_value': self.right_vr_data['inputs']['trigger'],
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
            
            print(f"\n🎯 === DUAL ARM VR Bridge 상태 ===")
            
            # 왼팔 상태
            print(f"🖐 LEFT ARM:")
            print(f"  캘리브레이션: {'✅' if self.left_vr_data['hand_pose']['calibrated'] else '❌'}")
            print(f"  트리거: {self.left_vr_data['inputs']['trigger']:.2f}")
            print(f"  조인트: {[f'{j:.2f}' for j in self.left_robot_joints]}")
            
            if self.left_vr_data['hand_pose']['calibrated']:
                left_pos, left_ori = self.get_vr_deltas(self.left_vr_data)
                if left_pos is not None and left_ori is not None:
                    print(f"  Pitch: {left_ori[1]:+.3f} → J4: {self.left_robot_joints[3]:+.3f}")
            
            # 오른팔 상태
            print(f"🖐 RIGHT ARM:")
            print(f"  캘리브레이션: {'✅' if self.right_vr_data['hand_pose']['calibrated'] else '❌'}")
            print(f"  트리거: {self.right_vr_data['inputs']['trigger']:.2f}")
            print(f"  조인트: {[f'{j:.2f}' for j in self.right_robot_joints]}")
            
            if self.right_vr_data['hand_pose']['calibrated']:
                right_pos, right_ori = self.get_vr_deltas(self.right_vr_data)
                if right_pos is not None and right_ori is not None:
                    print(f"  Pitch: {right_ori[1]:+.3f} → J4: {self.right_robot_joints[3]:+.3f}")
            
            print(f"⚡ 제어 주파수: {self.stats['control_frequency']:.1f}Hz")
            print(f"🌐 MuJoCo 클라이언트: {len(self.clients)}개")

if __name__ == "__main__":
    bridge = DualArmIntuitiveVRBridge()
    
    print("\n🎯 === DUAL ARM VR 제어 시스템 ===")
    print("🖐 왼팔: 왼쪽 VR 컨트롤러")
    print("🖐 오른팔: 오른쪽 VR 컨트롤러")
    print("🤏 Joint4 직관적 제어 (손목 회전)")
    print("🎮 트리거 → 그리퍼 제어")
    print("🔄 A+B 버튼 → 재캘리브레이션")
    print("📍 양팔 독립 제어 + 하나의 소켓 통신")
    
    try:
        while not rospy.is_shutdown():
            try:
                key = input().strip().lower()
                
                if key == 'c':  # 양팔 재캘리브레이션
                    bridge.recalibrate('left')
                    bridge.recalibrate('right')
                elif key == 'l':  # 왼팔만 재캘리브레이션
                    bridge.recalibrate('left')
                elif key == 'r':  # 오른팔만 재캘리브레이션
                    bridge.recalibrate('right')
                elif key == 'q':
                    break
                    
            except (EOFError, KeyboardInterrupt):
                break
                
    except:
        pass
    
    print("🏁 Dual Arm VR Bridge 시스템 종료")