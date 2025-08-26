#!/usr/bin/env python3
"""
🎮 Meta Quest 2 VR → Dual-Arm MuJoCo Bridge (Docker/ROS1)

이 파일은 Meta Quest 2 VR 헤드셋의 양쪽 컨트롤러 포즈를 두 개의 OpenManipulator-X 
로봇(왼쪽, 오른쪽)의 조인트 각도로 변환하는 핵심 VR 브릿지입니다.

주요 특징:
🎯 왼쪽 컨트롤러 → 왼쪽 로봇 (기존 기능 유지)
🎯 오른쪽 컨트롤러 → 오른쪽 로봇 (새로 추가)
🔄 Offset-based Control Method (양쪽 모두)
📡 Socket을 통해 Host로 양팔 조인트 값 전송
⚙️ 개별 캘리브레이션 및 제어
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

class DualArmVRBridge:
    def __init__(self):
        rospy.init_node('dual_arm_vr_bridge')
        
        print("🤖 양팔 VR Bridge V1.0 시작")
        print("🤏 양쪽 컨트롤러 → 양쪽 OpenManipulator-X")
        
        # 소켓 서버 설정
        self.setup_socket_server()
        
        # 양팔 VR 데이터 저장 구조
        self.vr_data = {
            'left_hand': {
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'initial_position': None,
                'initial_orientation': None,
                'calibrated': False,
                'pose_history': deque(maxlen=15),
                'orientation_history': deque(maxlen=20)
            },
            'right_hand': {
                'position': np.array([0.0, 0.0, 0.0]),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'initial_position': None,
                'initial_orientation': None,
                'calibrated': False,
                'pose_history': deque(maxlen=15),
                'orientation_history': deque(maxlen=20)
            },
            'inputs': {
                'left_trigger': 0.0,
                'left_button_upper': False,
                'left_button_lower': False,
                'right_trigger': 0.0,
                'right_button_upper': False,
                'right_button_lower': False
            }
        }
        
        # 양팔 로봇 조인트 상태
        self.robot_joints = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }
        self.target_joints = {
            'left': [0.0, 0.0, 0.0, 0.0],
            'right': [0.0, 0.0, 0.0, 0.0]
        }
        self.gripper_values = {
            'left': -0.01,
            'right': -0.01
        }
        
        # 안전 파라미터 (양쪽 동일)
        self.safety_params = {
            'max_joint_speed': 0.08,
            'position_scale': np.array([0.7, 0.7, 0.7]),
            'smooth_factor': 0.08,
            'z_axis_gain': 0.4,
            'joint1_gain': 0.55,
            'safety_margin': 0.9
        }
        
        # Z축 매핑 (양쪽 동일)
        self.z_axis_mapping = {
            'down_threshold': -0.08,
            'up_threshold': 0.08,
            'joint2_down_gain': 0.35,
            'joint3_down_gain': 0.25,
            'joint2_up_gain': -0.25,
            'joint3_up_gain': -0.35,
        }
        
        # Joint4 직관적 매핑 (양쪽 동일)
        self.joint4_mapping = {
            'pitch_sensitivity': 1.2,
            'roll_sensitivity': 0.3,
            'yaw_sensitivity': 0.2,
            'direct_influence': 0.7,
            'smoothing_factor': 0.15,
            'deadzone': 0.05,
            'max_change_rate': 0.12
        }
        
        # 오른쪽 팔 대칭 매핑 오프셋 (오른쪽은 왼쪽의 미러링)
        # MuJoCo 시뮬레이션: Y축 반전 비활성화 (1.0)
        # 실제 로봇: Y축 반전 활성화 (-1.0)
        USE_MUJOCO_MODE = False  # True: MuJoCo용, False: 실제 로봇용 [실물 로봇 모드]
        
        self.right_arm_mirror_offset = {
            'joint1_multiplier': 1.0,  # Joint1 반전 제거 (양팔 동일하게)
            'position_y_multiplier': 1.0  # Y축 반전 제거 (양팔 동일하게)
        }
        
        # 매핑 데이터 로드 (양쪽 공통 사용)
        self.load_mapping_data()
        self.build_mapping_trees()
        
        # 필터 설정
        self.setup_filters()
        
        # 성능 통계
        self.stats = {
            'left_control_count': 0,
            'right_control_count': 0,
            'control_frequency': 0.0,
            'dual_sync_rate': 0.0
        }
        
        # ROS 설정
        self.setup_ros_topics()
        
        # 제어 루프 시작 (양팔 통합)
        self.control_thread = threading.Thread(target=self.dual_arm_control_loop, daemon=True)
        self.control_thread.start()
        
        # 디버그 스레드
        self.debug_thread = threading.Thread(target=self.debug_loop, daemon=True)
        self.debug_thread.start()
        
        print("✅ 양팔 VR Bridge 초기화 완료")
        print("🤏 왼쪽/오른쪽 컨트롤러 → 왼쪽/오른쪽 로봇")
    
    def load_mapping_data(self):
        """매핑 데이터 로드 (기존과 동일, 양팔 공통 사용)"""
        print("📁 매핑 데이터 로드 중...")
        
        # 기존 test3.py와 동일한 매핑 데이터
        base_data = [
            {"vr_pos": [0.001, -0.013, -0.0003], "vr_ori": [-0.022, 0.043, -0.054], "joints": [0.0, 0.0, 0.0, 0.0]},
            {"vr_pos": [0.019, -0.003, -0.021], "vr_ori": [0.149, 0.689, 0.109], "joints": [0.0, 0.0, 0.0, 0.5]},
            {"vr_pos": [-0.036, -0.012, 0.048], "vr_ori": [-0.037, -0.672, -0.056], "joints": [0.0, 0.0, 0.0, -0.5]},
        ]
        
        # Orientation 중심 증강 데이터
        orientation_samples = []
        for pos in [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]:
            for pitch in np.linspace(-1.2, 1.2, 9):
                joint4 = pitch * 0.7
                sample = {
                    "vr_pos": pos,
                    "vr_ori": [0.0, pitch, 0.0],
                    "joints": [0.0, 0.0, 0.0, joint4]
                }
                orientation_samples.append(sample)
        
        self.mapping_data = base_data + orientation_samples
        print(f"✅ {len(self.mapping_data)}개 매핑 데이터 로드 완료")
    
    def build_mapping_trees(self):
        """매핑 트리 구성 (기존과 동일)"""
        positions = np.array([sample['vr_pos'] for sample in self.mapping_data])
        self.position_tree = cKDTree(positions)
        
        orientations = np.array([sample['vr_ori'] for sample in self.mapping_data])
        self.orientation_tree = cKDTree(orientations)
        
        # Pitch 중심 트리
        pitch_features = []
        for sample in self.mapping_data:
            pitch_weighted = [
                sample['vr_ori'][0] * 0.3,
                sample['vr_ori'][1] * 1.5,
                sample['vr_ori'][2] * 0.2
            ]
            pitch_features.append(pitch_weighted)
        self.pitch_tree = cKDTree(np.array(pitch_features))
    
    def setup_filters(self):
        """필터 설정 (양팔 공통)"""
        self.filter_freq = 5.0
        self.filter_order = 3
        self.joint4_filter_freq = 10.0
        self.joint4_filter_order = 2
        
        nyquist = 120.0 / 2
        normalized_freq = self.filter_freq / nyquist
        self.filter_b, self.filter_a = butter(self.filter_order, normalized_freq, btype='low')
        
        j4_normalized_freq = self.joint4_filter_freq / nyquist
        self.j4_filter_b, self.j4_filter_a = butter(self.joint4_filter_order, j4_normalized_freq, btype='low')
        
        # 양팔 별도 필터 히스토리
        self.filter_history = {
            'left': {'joint_targets': deque(maxlen=30), 'joint4_targets': deque(maxlen=15)},
            'right': {'joint_targets': deque(maxlen=30), 'joint4_targets': deque(maxlen=15)}
        }
    
    def intuitive_joint4_control(self, vr_pos_delta, vr_ori_delta, arm_side='left'):
        """직관적 Joint4 제어 (기존 알고리즘, 양팔 적용)"""
        # Joint 1-3 계산 (기존과 동일)
        joint1 = np.arctan2(vr_pos_delta[1], vr_pos_delta[0] + 0.25) * self.safety_params['joint1_gain']
        
        # 양팔 동일하게 처리 (미러링 제거)
        # if arm_side == 'right':
        #     joint1 *= self.right_arm_mirror_offset['joint1_multiplier']
        
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
        
        # Joint4: Orientation 기반 제어 (기존과 동일)
        pitch = vr_ori_delta[1]
        roll = vr_ori_delta[0]
        yaw = vr_ori_delta[2]
        
        if abs(pitch) < self.joint4_mapping['deadzone']:
            pitch = 0.0
        
        joint4_from_pitch = pitch * self.joint4_mapping['pitch_sensitivity']
        joint4_from_roll = roll * self.joint4_mapping['roll_sensitivity']
        joint4_from_yaw = yaw * self.joint4_mapping['yaw_sensitivity']
        
        # Orientation KD-Tree 보간
        pitch_weighted_query = [roll * 0.3, pitch * 1.5, yaw * 0.2]
        ori_distances, ori_indices = self.pitch_tree.query(pitch_weighted_query, k=min(6, len(self.mapping_data)))
        
        if isinstance(ori_distances, float):
            ori_distances = [ori_distances]
            ori_indices = [ori_indices]
        
        ori_weights = 1.0 / (np.array(ori_distances) + 1e-6)
        ori_weights = ori_weights / np.sum(ori_weights)
        joint4_from_tree = sum(ori_weights[i] * self.mapping_data[idx]['joints'][3] for i, idx in enumerate(ori_indices))
        
        # 최종 Joint4 결합
        joint4_direct = joint4_from_pitch + joint4_from_roll * 0.3 + joint4_from_yaw * 0.2
        joint4 = (
            self.joint4_mapping['direct_influence'] * joint4_direct +
            (1 - self.joint4_mapping['direct_influence']) * joint4_from_tree
        )
        
        joint4 = np.clip(joint4, -1.0, 1.0)
        
        return [joint1, joint2, joint3, joint4]
    
    def apply_filter(self, target_joints, arm_side='left'):
        """필터 적용 (양팔 별도)"""
        history = self.filter_history[arm_side]
        history['joint_targets'].append(target_joints)
        history['joint4_targets'].append(target_joints[3])
        
        if len(history['joint_targets']) < 8:
            return target_joints
        
        recent_targets = np.array(list(history['joint_targets'])[-15:])
        recent_j4 = np.array(list(history['joint4_targets'])[-10:])
        
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
    
    def get_vr_deltas(self, arm_side='left'):
        """VR 델타 계산 (양팔 별도)"""
        hand_data = self.vr_data[f'{arm_side}_hand']
        
        if not hand_data['calibrated']:
            return None, None
        
        current_pos = hand_data['position']
        initial_pos = hand_data['initial_position']
        position_delta = (current_pos - initial_pos) * self.safety_params['position_scale']
        
        # 양팔 동일하게 처리 (Y축 반전 제거)
        # if arm_side == 'right':
        #     position_delta[1] *= self.right_arm_mirror_offset['position_y_multiplier']
        
        position_delta = np.clip(position_delta, -0.3, 0.3)
        
        current_ori = hand_data['orientation']
        initial_ori = hand_data['initial_orientation']
        
        current_euler = tf_trans.euler_from_quaternion(current_ori)
        initial_euler = tf_trans.euler_from_quaternion(initial_ori)
        orientation_delta = np.array(current_euler) - np.array(initial_euler)
        orientation_delta = np.clip(orientation_delta, -2.0, 2.0)
        
        return position_delta, orientation_delta
    
    def dual_arm_control_loop(self):
        """양팔 통합 제어 루프"""
        rate = rospy.Rate(120)
        
        while not rospy.is_shutdown():
            loop_start_time = time.time()
            
            # 왼쪽 팔 제어
            if self.vr_data['left_hand']['calibrated']:
                vr_pos_delta, vr_ori_delta = self.get_vr_deltas('left')
                if vr_pos_delta is not None and vr_ori_delta is not None:
                    try:
                        raw_target = self.intuitive_joint4_control(vr_pos_delta, vr_ori_delta, 'left')
                        filtered_target = self.apply_filter(raw_target, 'left')
                        
                        # 부드러운 업데이트
                        for i in range(4):
                            joint_error = filtered_target[i] - self.robot_joints['left'][i]
                            if i == 3:
                                max_change = self.joint4_mapping['max_change_rate']
                                smooth_factor = self.joint4_mapping['smoothing_factor']
                            else:
                                max_change = self.safety_params['max_joint_speed']
                                smooth_factor = self.safety_params['smooth_factor']
                            
                            joint_error = np.clip(joint_error, -max_change, max_change)
                            self.robot_joints['left'][i] += joint_error * smooth_factor
                            
                            # 안전 제한
                            if i == 3:
                                self.robot_joints['left'][i] = np.clip(self.robot_joints['left'][i], -1.2, 1.2)
                            else:
                                if abs(self.robot_joints['left'][i]) > 1.3:
                                    self.robot_joints['left'][i] = np.sign(self.robot_joints['left'][i]) * 1.3
                        
                        self.stats['left_control_count'] += 1
                    except Exception as e:
                        rospy.logwarn(f"왼쪽 제어 오류: {e}")
            
            # 오른쪽 팔 제어
            if self.vr_data['right_hand']['calibrated']:
                vr_pos_delta, vr_ori_delta = self.get_vr_deltas('right')
                if vr_pos_delta is not None and vr_ori_delta is not None:
                    try:
                        raw_target = self.intuitive_joint4_control(vr_pos_delta, vr_ori_delta, 'right')
                        filtered_target = self.apply_filter(raw_target, 'right')
                        
                        # 부드러운 업데이트
                        for i in range(4):
                            joint_error = filtered_target[i] - self.robot_joints['right'][i]
                            if i == 3:
                                max_change = self.joint4_mapping['max_change_rate']
                                smooth_factor = self.joint4_mapping['smoothing_factor']
                            else:
                                max_change = self.safety_params['max_joint_speed']
                                smooth_factor = self.safety_params['smooth_factor']
                            
                            joint_error = np.clip(joint_error, -max_change, max_change)
                            self.robot_joints['right'][i] += joint_error * smooth_factor
                            
                            # 안전 제한
                            if i == 3:
                                self.robot_joints['right'][i] = np.clip(self.robot_joints['right'][i], -1.2, 1.2)
                            else:
                                if abs(self.robot_joints['right'][i]) > 1.3:
                                    self.robot_joints['right'][i] = np.sign(self.robot_joints['right'][i]) * 1.3
                        
                        self.stats['right_control_count'] += 1
                    except Exception as e:
                        rospy.logwarn(f"오른쪽 제어 오류: {e}")
            
            # 그리퍼 제어
            self.update_gripper_control()
            
            # MuJoCo로 데이터 전송
            self.send_to_mujoco()
            
            loop_time = time.time() - loop_start_time
            self.stats['control_frequency'] = 1.0 / max(loop_time, 0.001)
            
            rate.sleep()
    
    def update_gripper_control(self):
        """그리퍼 제어 (양팔)"""
        # 왼쪽 그리퍼
        left_trigger = self.vr_data['inputs']['left_trigger']
        self.gripper_values['left'] = -0.01 + (left_trigger * 0.029)
        if self.vr_data['inputs']['left_button_upper']:
            self.gripper_values['left'] = 0.019
        
        # 오른쪽 그리퍼
        right_trigger = self.vr_data['inputs']['right_trigger']
        self.gripper_values['right'] = -0.01 + (right_trigger * 0.029)
        if self.vr_data['inputs']['right_button_upper']:
            self.gripper_values['right'] = 0.019
    
    def send_to_mujoco(self):
        """MuJoCo로 양팔 데이터 전송"""
        if self.clients:
            data = {
                'left_arm': {
                    'joint_angles': self.robot_joints['left'],
                    'gripper': self.gripper_values['left']
                },
                'right_arm': {
                    'joint_angles': self.robot_joints['right'],
                    'gripper': self.gripper_values['right']
                },
                'vr_status': {
                    'left_calibrated': self.vr_data['left_hand']['calibrated'],
                    'right_calibrated': self.vr_data['right_hand']['calibrated'],
                    'left_trigger': self.vr_data['inputs']['left_trigger'],
                    'right_trigger': self.vr_data['inputs']['right_trigger'],
                    'left_buttons': {
                        'upper': self.vr_data['inputs']['left_button_upper'],
                        'lower': self.vr_data['inputs']['left_button_lower']
                    },
                    'right_buttons': {
                        'upper': self.vr_data['inputs']['right_button_upper'],
                        'lower': self.vr_data['inputs']['right_button_lower']
                    }
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
        """ROS 토픽 설정 (양쪽 컨트롤러)"""
        # 왼쪽 컨트롤러 (기존)
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, 
                        lambda msg: self.hand_pose_callback(msg, 'left'))
        
        # 오른쪽 컨트롤러 (새로 추가)
        rospy.Subscriber('/q2r_right_hand_pose', PoseStamped, 
                        lambda msg: self.hand_pose_callback(msg, 'right'))
        
        # VR 입력 (양쪽)
        try:
            from quest2ros.msg import OVR2ROSInputs
            rospy.Subscriber('/q2r_left_hand_inputs', OVR2ROSInputs, 
                            lambda msg: self.input_callback(msg, 'left'))
            rospy.Subscriber('/q2r_right_hand_inputs', OVR2ROSInputs, 
                            lambda msg: self.input_callback(msg, 'right'))
            print("✅ 양쪽 VR 입력 토픽 구독됨")
        except ImportError:
            print("⚠️ OVR2ROSInputs 메시지 없음")
        
        print("✅ ROS 토픽 설정 완료")
    
    def hand_pose_callback(self, msg, arm_side='left'):
        """VR 손 Pose 콜백 (양팔 별도 처리)"""
        hand_data = self.vr_data[f'{arm_side}_hand']
        
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
        hand_data['pose_history'].append(pose_data)
        hand_data['orientation_history'].append(current_orientation)
        
        # 스무딩 (기존 알고리즘과 동일)
        if len(hand_data['pose_history']) >= 8:
            recent_poses = list(hand_data['pose_history'])[-12:]
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
            
            recent_ori_weights = weights[-6:]
            recent_ori_weights = recent_ori_weights / np.sum(recent_ori_weights)
            smoothed_orientation = np.average(orientations[-6:], axis=0, weights=recent_ori_weights)
            smoothed_orientation = smoothed_orientation / np.linalg.norm(smoothed_orientation)
        else:
            smoothed_position = current_position
            smoothed_orientation = current_orientation
        
        hand_data['position'] = smoothed_position
        hand_data['orientation'] = smoothed_orientation
        
        # 캘리브레이션
        if not hand_data['calibrated']:
            hand_data['initial_position'] = smoothed_position.copy()
            hand_data['initial_orientation'] = smoothed_orientation.copy()
            hand_data['calibrated'] = True
            print(f"🖐 {arm_side.upper()} VR 컨트롤러 캘리브레이션 완료")
    
    def input_callback(self, msg, arm_side='left'):
        """VR 입력 콜백 (양팔 별도)"""
        try:
            input_prefix = f'{arm_side}_'
            
            if hasattr(msg, 'trigger'):
                self.vr_data['inputs'][f'{input_prefix}trigger'] = msg.trigger
            if hasattr(msg, 'button_upper'):
                self.vr_data['inputs'][f'{input_prefix}button_upper'] = msg.button_upper
            if hasattr(msg, 'button_lower'):
                self.vr_data['inputs'][f'{input_prefix}button_lower'] = msg.button_lower
            
            # 양쪽 A+B 버튼 동시 누르면 해당 팔 재캘리브레이션
            if (self.vr_data['inputs'][f'{input_prefix}button_upper'] and 
                self.vr_data['inputs'][f'{input_prefix}button_lower']):
                self.recalibrate(arm_side)
                
        except Exception as e:
            rospy.logwarn(f"{arm_side} 입력 처리 오류: {e}")
    
    def recalibrate(self, arm_side='left'):
        """재캘리브레이션 (팔별 개별)"""
        hand_data = self.vr_data[f'{arm_side}_hand']
        if hand_data['position'] is not None:
            hand_data['initial_position'] = hand_data['position'].copy()
            hand_data['initial_orientation'] = hand_data['orientation'].copy()
            hand_data['pose_history'].clear()
            hand_data['orientation_history'].clear()
            print(f"🔄 {arm_side.upper()} VR 컨트롤러 재캘리브레이션 완료!")
    
    def debug_loop(self):
        """디버그 출력"""
        while not rospy.is_shutdown():
            time.sleep(4.0)
            
            print(f"\n🤖 === 양팔 VR Bridge 상태 ===")
            print(f"🖐 왼쪽 캘리브레이션: {'✅' if self.vr_data['left_hand']['calibrated'] else '❌'}")
            print(f"🖐 오른쪽 캘리브레이션: {'✅' if self.vr_data['right_hand']['calibrated'] else '❌'}")
            print(f"🎮 트리거: L={self.vr_data['inputs']['left_trigger']:.2f}, R={self.vr_data['inputs']['right_trigger']:.2f}")
            print(f"🤖 왼쪽 조인트: J1={self.robot_joints['left'][0]:.3f}, J2={self.robot_joints['left'][1]:.3f}, J3={self.robot_joints['left'][2]:.3f}, J4={self.robot_joints['left'][3]:.3f}")
            print(f"🤖 오른쪽 조인트: J1={self.robot_joints['right'][0]:.3f}, J2={self.robot_joints['right'][1]:.3f}, J3={self.robot_joints['right'][2]:.3f}, J4={self.robot_joints['right'][3]:.3f}")
            print(f"📊 제어 횟수: L={self.stats['left_control_count']}, R={self.stats['right_control_count']}")
            print(f"⚡ 제어 주파수: {self.stats['control_frequency']:.1f}Hz")

if __name__ == "__main__":
    bridge = DualArmVRBridge()
    
    print("\n🤖 === 양팔 VR Bridge 시스템 V1.0 ===")
    print("🎯 왼쪽/오른쪽 컨트롤러 → 왼쪽/오른쪽 OpenManipulator-X")
    print("🎮 트리거 → 그리퍼 제어")
    print("🔄 각각 A+B 버튼 → 개별 재캘리브레이션")
    print("🎯 왼쪽: 기존 로봇 (모터 ID 11~15)")
    print("🎯 오른쪽: 새 로봇 (모터 ID 21~25 등)")
    
    try:
        while not rospy.is_shutdown():
            try:
                key = input().strip().lower()
                
                if key == 'cl':
                    bridge.recalibrate('left')
                elif key == 'cr':
                    bridge.recalibrate('right')
                elif key == 'c':
                    bridge.recalibrate('left')
                    bridge.recalibrate('right')
                elif key == 'r':
                    bridge.robot_joints['left'] = [0.0, 0.0, 0.0, 0.0]
                    bridge.robot_joints['right'] = [0.0, 0.0, 0.0, 0.0]
                    bridge.gripper_values['left'] = -0.01
                    bridge.gripper_values['right'] = -0.01
                    print("🔄 양쪽 로봇 리셋됨")
                elif key == 'q':
                    break
                    
            except (EOFError, KeyboardInterrupt):
                break
                
    except:
        pass
    
    print("🏁 양팔 VR Bridge 시스템 종료")