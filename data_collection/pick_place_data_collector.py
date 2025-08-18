#!/usr/bin/env python3
"""
Pick & Place 데이터 수집 시스템
- 정확한 End-Effector 위치 계산
- 연속 궤적 데이터 수집
- 강화학습 적합 데이터셋 생성
"""

import rospy
import numpy as np
import json
import time
import threading
from collections import deque
from geometry_msgs.msg import PoseStamped
import mujoco

class PickPlaceDataCollector:
    def __init__(self):
        rospy.init_node('pick_place_data_collector')
        
        print("🎯 Pick & Place 데이터 수집 시스템 시작")
        
        # MuJoCo 모델 로드 (Forward Kinematics용)
        try:
            self.model = mujoco.MjModel.from_xml_path('scene.xml')
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo 모델 로드 완료")
        except Exception as e:
            print(f"❌ MuJoCo 모델 로드 실패: {e}")
            raise
        
        # 데이터 수집 상태
        self.is_collecting = False
        self.current_episode = {
            'episode_id': 0,
            'task_type': 'pick_and_place',
            'start_time': None,
            'trajectory': [],
            'success': False,
            'objects': []
        }
        self.episodes = []
        
        # VR 및 로봇 상태
        self.robot_joints = [0.0, 0.0, 0.0, 0.0]
        self.gripper_value = -0.01
        self.vr_calibrated = False
        self.vr_initial_pose = None
        
        # ROS 토픽 설정
        self.setup_ros_topics()
        
        print("✅ Pick & Place 데이터 수집 시스템 준비 완료")
    
    def get_accurate_end_effector_pose(self):
        """정확한 End-Effector 위치 계산"""
        # 현재 관절 각도로 Forward Kinematics
        self.data.qpos[:4] = self.robot_joints
        mujoco.mj_forward(self.model, self.data)
        
        try:
            # 그리퍼 중심점 계산
            left_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_left_link')
            right_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_right_link')
            
            if left_gripper_id >= 0 and right_gripper_id >= 0:
                left_pos = self.data.xpos[left_gripper_id]
                right_pos = self.data.xpos[right_gripper_id]
                gripper_center = (left_pos + right_pos) / 2.0
                
                # 그리퍼 앞쪽 끝 (실제 작업점)
                link5_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link5')
                link5_rotation = self.data.xmat[link5_id].reshape(3, 3)
                gripper_tip_offset = link5_rotation @ np.array([0.02, 0, 0])
                actual_ee_pos = gripper_center + gripper_tip_offset
                
                # Orientation
                ee_quat = self.data.xquat[link5_id].copy()
                
                return actual_ee_pos.tolist(), ee_quat.tolist()
        except Exception as e:
            print(f"⚠️ End-effector 계산 오류: {e}")
            
        return [0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]
    
    def setup_ros_topics(self):
        """ROS 토픽 설정"""
        rospy.Subscriber('/q2r_left_hand_pose', PoseStamped, self.vr_pose_callback)
        print("✅ VR 토픽 구독 설정")
    
    def vr_pose_callback(self, msg):
        """VR Pose 콜백 - 기존 test3.py 로직 사용"""
        # 여기에 test3.py의 VR 처리 로직 통합
        pass
    
    def start_episode(self, task_type="pick_and_place"):
        """에피소드 시작"""
        if self.is_collecting:
            print("⚠️ 이미 수집 중입니다")
            return
        
        self.current_episode = {
            'episode_id': len(self.episodes) + 1,
            'task_type': task_type,
            'start_time': time.time(),
            'trajectory': [],
            'success': False,
            'objects': [],
            'metadata': {
                'gripper_open_threshold': 0.01,
                'gripper_close_threshold': -0.005,
                'success_criteria': 'object_at_target_position'
            }
        }
        
        self.is_collecting = True
        print(f"🎬 에피소드 {self.current_episode['episode_id']} 시작 - {task_type}")
    
    def record_frame(self):
        """현재 프레임 기록"""
        if not self.is_collecting:
            return
        
        # 정확한 End-Effector 위치 계산
        ee_pos, ee_quat = self.get_accurate_end_effector_pose()
        
        # VR 델타 계산 (test3.py 로직 필요)
        vr_pos_delta = [0, 0, 0]  # TODO: VR 처리 로직 추가
        vr_ori_delta = [0, 0, 0]
        
        frame_data = {
            'timestamp': time.time() - self.current_episode['start_time'],
            'step_id': len(self.current_episode['trajectory']),
            
            # VR 데이터
            'vr_position_delta': vr_pos_delta,
            'vr_orientation_delta': vr_ori_delta,
            
            # 로봇 상태 (Joint Space)
            'joint_angles': self.robot_joints.copy(),
            'joint_velocities': [0, 0, 0, 0],  # TODO: 속도 계산
            
            # End-Effector (Task Space) - 정확한 위치!
            'end_effector_position': ee_pos,
            'end_effector_quaternion': ee_quat,
            'end_effector_velocity': [0, 0, 0],  # TODO: 속도 계산
            
            # 그리퍼 상태
            'gripper_position': self.gripper_value,
            'gripper_state': 'open' if self.gripper_value > 0.01 else 'closed',
            
            # 작업 정보
            'action_type': self.detect_action_type(),
            'contact_detected': False,  # TODO: 접촉 감지
            'object_grasped': self.is_object_grasped(),
        }
        
        self.current_episode['trajectory'].append(frame_data)
    
    def detect_action_type(self):
        """현재 동작 유형 감지"""
        # 간단한 휴리스틱
        if len(self.current_episode['trajectory']) < 5:
            return 'approaching'
        
        recent_frames = self.current_episode['trajectory'][-5:]
        
        # 그리퍼 상태 변화 감지
        gripper_states = [f['gripper_state'] for f in recent_frames]
        if 'closed' in gripper_states and gripper_states[-1] == 'closed':
            return 'grasping'
        elif 'open' in gripper_states and gripper_states[-1] == 'open':
            return 'releasing'
        
        # 이동 감지
        positions = [f['end_effector_position'] for f in recent_frames]
        if len(positions) >= 2:
            distance = np.linalg.norm(np.array(positions[-1]) - np.array(positions[0]))
            if distance > 0.01:  # 1cm 이상 이동
                return 'moving'
        
        return 'holding'
    
    def is_object_grasped(self):
        """객체 파지 여부 감지 (간단한 버전)"""
        return self.gripper_value < -0.005
    
    def end_episode(self, success=True):
        """에피소드 종료"""
        if not self.is_collecting:
            print("⚠️ 수집 중이 아닙니다")
            return
        
        self.current_episode['success'] = success
        self.current_episode['end_time'] = time.time()
        self.current_episode['duration'] = self.current_episode['end_time'] - self.current_episode['start_time']
        
        # 성공률 계산
        success_rate = len([e for e in self.episodes if e['success']]) / max(len(self.episodes), 1)
        
        print(f"🏁 에피소드 {self.current_episode['episode_id']} 종료")
        print(f"   성공: {'✅' if success else '❌'}")
        print(f"   지속시간: {self.current_episode['duration']:.1f}초")
        print(f"   프레임 수: {len(self.current_episode['trajectory'])}")
        print(f"   전체 성공률: {success_rate:.1%}")
        
        self.episodes.append(self.current_episode.copy())
        self.is_collecting = False
    
    def save_dataset(self, filename=None):
        """데이터셋 저장"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"pick_place_dataset_{timestamp}.json"
        
        dataset = {
            'metadata': {
                'collection_method': 'vr_teleoperation',
                'task_type': 'pick_and_place',
                'total_episodes': len(self.episodes),
                'successful_episodes': len([e for e in self.episodes if e['success']]),
                'success_rate': len([e for e in self.episodes if e['success']]) / max(len(self.episodes), 1),
                'collection_timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'end_effector_accuracy': 'verified_gripper_position',
                'robot_model': 'OpenManipulator-X',
                'control_method': 'vr_teleoperation'
            },
            'episodes': self.episodes
        }
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 데이터셋 저장 완료: {filename}")
        print(f"   총 에피소드: {len(self.episodes)}개")
        print(f"   성공률: {dataset['metadata']['success_rate']:.1%}")
        
        return filename

    def run_data_collection(self):
        """데이터 수집 실행"""
        print("\n🎯 Pick & Place 데이터 수집 시작")
        print("=" * 50)
        print("📝 사용법:")
        print("   's' - 에피소드 시작")
        print("   'e' - 에피소드 종료 (성공)")
        print("   'f' - 에피소드 종료 (실패)")
        print("   'v' - 현재 상태 확인")
        print("   'save' - 데이터셋 저장")
        print("   'q' - 종료")
        print("=" * 50)
        
        # 실시간 데이터 기록 스레드
        def recording_loop():
            rate = rospy.Rate(30)  # 30Hz
            while not rospy.is_shutdown():
                if self.is_collecting:
                    self.record_frame()
                rate.sleep()
        
        recording_thread = threading.Thread(target=recording_loop, daemon=True)
        recording_thread.start()
        
        try:
            while not rospy.is_shutdown():
                try:
                    command = input().strip().lower()
                    
                    if command == 's':
                        self.start_episode()
                    elif command == 'e':
                        self.end_episode(success=True)
                    elif command == 'f':
                        self.end_episode(success=False)
                    elif command == 'v':
                        self.print_status()
                    elif command == 'save':
                        self.save_dataset()
                    elif command == 'q':
                        break
                        
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("🏁 데이터 수집 시스템 종료")
    
    def print_status(self):
        """현재 상태 출력"""
        print(f"\n📊 현재 상태:")
        print(f"   수집 중: {'🔴' if self.is_collecting else '⚫'}")
        print(f"   총 에피소드: {len(self.episodes)}개")
        if self.is_collecting:
            print(f"   현재 에피소드: {self.current_episode['episode_id']}")
            print(f"   현재 프레임: {len(self.current_episode['trajectory'])}개")
        print(f"   관절 각도: {self.robot_joints}")
        print(f"   그리퍼: {self.gripper_value:.3f}")

if __name__ == "__main__":
    try:
        collector = PickPlaceDataCollector()
        collector.run_data_collection()
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()