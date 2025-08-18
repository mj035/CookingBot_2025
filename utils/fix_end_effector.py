#!/usr/bin/env python3
"""
End-Effector 위치 정확성 검증 및 수정
"""

import mujoco
import numpy as np

def get_accurate_end_effector_pose(model, data):
    """정확한 End-Effector 위치 계산"""
    
    # 방법 1: 그리퍼 중심점 계산 (더 정확)
    try:
        left_gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_left_link')
        right_gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_right_link')
        
        if left_gripper_id >= 0 and right_gripper_id >= 0:
            left_pos = data.xpos[left_gripper_id]
            right_pos = data.xpos[right_gripper_id]
            
            # 두 그리퍼 중심점
            gripper_center = (left_pos + right_pos) / 2.0
            
            # 그리퍼 앞쪽 끝 (실제 작업점)
            link5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link5')
            link5_rotation = data.xmat[link5_id].reshape(3, 3)
            
            # X축 방향으로 그리퍼 길이만큼 앞쪽 (약 2cm)
            gripper_tip_offset = link5_rotation @ np.array([0.02, 0, 0])
            actual_ee_pos = gripper_center + gripper_tip_offset
            
            # 그리퍼 orientation (link5와 동일)
            ee_quat = data.xquat[link5_id].copy()
            
            return actual_ee_pos, ee_quat, "gripper_center_method"
            
    except Exception as e:
        print(f"그리퍼 중심점 계산 실패: {e}")
    
    # 방법 2: Link5 + 오프셋 (대안)
    try:
        link5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link5')
        if link5_id >= 0:
            link5_pos = data.xpos[link5_id]
            link5_quat = data.xquat[link5_id]
            link5_rotation = data.xmat[link5_id].reshape(3, 3)
            
            # URDF 기준: gripper_left_link는 link5 + [0.0817, 0.021, 0]
            # 실제 작업점은 그리퍼 끝이므로 조금 더 앞쪽
            ee_offset = link5_rotation @ np.array([0.1017, 0, 0])  # 0.0817 + 0.02
            actual_ee_pos = link5_pos + ee_offset
            
            return actual_ee_pos, link5_quat, "link5_offset_method"
            
    except Exception as e:
        print(f"Link5 오프셋 계산 실패: {e}")
    
    # 방법 3: 기존 end_effector_target (부정확하지만 fallback)
    try:
        ee_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_target')
        if ee_target_id >= 0:
            pos = data.xpos[ee_target_id]
            quat = data.xquat[ee_target_id]
            return pos, quat, "target_method_inaccurate"
    except:
        pass
    
    # 최후 방법: 기본값
    return np.array([0.3, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]), "default_fallback"

def verify_end_effector_accuracy():
    """End-Effector 정확도 검증"""
    print("🔍 End-Effector 위치 정확도 검증")
    
    try:
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)
        
        # 몇 가지 관절 각도로 테스트
        test_configs = [
            [0.0, 0.0, 0.0, 0.0],      # 홈 포지션
            [0.5, -0.5, 0.5, 0.0],     # 일반 포즈 1
            [-0.5, 0.5, -0.5, 0.5],    # 일반 포즈 2
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n📊 테스트 {i+1}: {config}")
            
            # 관절 각도 설정
            data.qpos[:4] = config
            mujoco.mj_forward(model, data)
            
            # 각 방법으로 End-Effector 위치 계산
            pos1, quat1, method1 = get_accurate_end_effector_pose(model, data)
            
            # 기존 방법 (부정확)
            try:
                ee_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_target')
                old_pos = data.xpos[ee_target_id]
                old_quat = data.xquat[ee_target_id]
            except:
                old_pos = np.array([0, 0, 0])
                old_quat = np.array([1, 0, 0, 0])
            
            # 차이 계산
            position_diff = np.linalg.norm(pos1 - old_pos)
            
            print(f"   정확한 EE ({method1}): [{pos1[0]:.4f}, {pos1[1]:.4f}, {pos1[2]:.4f}]")
            print(f"   기존 EE (부정확):     [{old_pos[0]:.4f}, {old_pos[1]:.4f}, {old_pos[2]:.4f}]")
            print(f"   위치 차이: {position_diff:.4f}m ({position_diff*100:.1f}cm)")
            
            if position_diff > 0.03:  # 3cm 이상 차이
                print(f"   ⚠️ 큰 오차! 수정 필요")
            else:
                print(f"   ✅ 허용 범위")
    
    except Exception as e:
        print(f"❌ 검증 실패: {e}")

if __name__ == "__main__":
    verify_end_effector_accuracy()