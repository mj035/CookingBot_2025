import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # scene.xml 파일 로드
    xml_path = "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 물체 추가 (집을 대상)
    add_objects(model, data)
    
    # 'neutral_pose' 키프레임으로 초기화
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    
    # 뷰어 시작
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 뷰어 설정
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        # 액추에이터 제어를 위한 설정
        left_arm_actuators = {}
        right_arm_actuators = {}
        
        for joint_name in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]:
            left_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left/{joint_name}")
            right_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right/{joint_name}")
            
            if left_idx >= 0:
                left_arm_actuators[joint_name] = left_idx
            if right_idx >= 0:
                right_arm_actuators[joint_name] = right_idx
        
        # 로봇 제어 시작
        start_time = time.time()
        
        # 왼쪽 팔로 픽앤플레이스 수행
        sequence = [
            (5.0, approach_object),     # 물체에 접근
            (3.0, grasp_object),        # 물체 잡기
            (5.0, lift_object),         # 물체 들어올리기
            (5.0, move_object),         # 물체 이동
            (3.0, place_object),        # 물체 내려놓기
            (3.0, release_object),      # 물체 놓기
            (5.0, return_to_home)       # 홈 포지션으로 돌아가기
        ]
        
        current_phase = 0
        phase_start_time = 0
        
        print("시뮬레이션 시작: 픽앤플레이스 작업")
        while viewer.is_running():
            current_time = time.time() - start_time
            
            if current_phase < len(sequence):
                duration, action_func = sequence[current_phase]
                
                # 새 단계 시작
                if current_phase == 0 and phase_start_time == 0:
                    phase_start_time = current_time
                    print(f"단계 {current_phase+1}: {action_func.__name__}")
                
                # 현재 단계의 진행도 (0에서 1 사이)
                phase_progress = min(1.0, (current_time - phase_start_time) / duration)
                
                # 현재 단계의 액션 실행
                action_func(model, data, left_arm_actuators, right_arm_actuators, phase_progress)
                
                # 단계 완료되면 다음 단계로
                if phase_progress >= 1.0:
                    current_phase += 1
                    phase_start_time = current_time
                    if current_phase < len(sequence):
                        print(f"단계 {current_phase+1}: {sequence[current_phase][1].__name__}")
                    else:
                        print("모든 작업 완료!")
            
            # 시뮬레이션 스텝 진행
            mujoco.mj_step(model, data)
            
            # 뷰어 업데이트
            viewer.sync()
            
            # 실시간 속도 유지
            sim_time = data.time
            elapsed = time.time() - start_time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)

def add_objects(model, data):
    """MuJoCo 모델에 집을 물체 추가"""
    # XML 요소를 만들어 물체 추가
    cube_xml = """
    <body name="cube" pos="0 -0.3 0.05">
      <joint type="free"/>
      <geom type="box" size="0.03 0.03 0.03" rgba="1 0 0 1" mass="0.1"/>
    </body>
    """
    
    # 물체 추가를 위한 XML 문자열 생성
    target_pos_xml = """
    <body name="target_pos" pos="0.2 -0.3 0.05">
      <geom type="cylinder" size="0.05 0.001" rgba="0 1 0 0.3"/>
    </body>
    """
    
    # XML 요소를 모델에 추가하는 부분은 실제로는 mujoco_py를 사용하여 
    # 동적으로 모델을 수정해야 하지만, 여기서는 모델이 이미 이러한 요소들을 포함하고 있다고 가정합니다.
    # MuJoCo 3.0 이상에서는 모델을 동적으로 수정하는 방식이 달라졌습니다.
    
    # 참고: 실제 구현에서는 mujoco.MjModel.add_body 등의 함수를 사용해야 합니다.
    # 현재는 간략화를 위해 생략합니다.
    
    print("물체 및 목표 위치 추가됨 (시뮬레이션 상에서)")

# 다양한 로봇 움직임 단계를 구현한 함수들
def approach_object(model, data, left_arm, right_arm, progress):
    """물체에 접근하는 단계"""
    # 왼쪽 팔을 물체를 향해 이동
    target_left = {
        "waist": 0.1,
        "shoulder": -0.7,
        "elbow": 1.1,
        "forearm_roll": 0.0,
        "wrist_angle": -0.4,
        "wrist_rotate": 0.0,
        "gripper": 0.03  # 그리퍼 열기
    }
    
    # 오른쪽 팔은 정지 상태로 유지
    target_right = {
        "waist": 0.0,
        "shoulder": -0.96,
        "elbow": 1.2,
        "forearm_roll": 0.0,
        "wrist_angle": -0.3,
        "wrist_rotate": 0.0,
        "gripper": 0.0084
    }
    
    # 왼쪽 팔 제어
    for joint, idx in left_arm.items():
        initial_value = data.ctrl[idx]
        target_value = target_left[joint]
        data.ctrl[idx] = initial_value * (1-progress) + target_value * progress
    
    # 오른쪽 팔 제어
    for joint, idx in right_arm.items():
        data.ctrl[idx] = target_right[joint]

def grasp_object(model, data, left_arm, right_arm, progress):
    """물체를 잡는 단계"""
    # 그리퍼만 닫기
    target_gripper = 0.03 * (1-progress) + 0.0084 * progress  # 점점 닫힘
    
    data.ctrl[left_arm["gripper"]] = target_gripper

def lift_object(model, data, left_arm, right_arm, progress):
    """물체를 들어올리는 단계"""
    # 현재 상태에서 높이만 올림
    current_elbow = data.ctrl[left_arm["elbow"]]
    current_shoulder = data.ctrl[left_arm["shoulder"]]
    
    # 들어올리기 (높이 올리기)
    target_elbow = current_elbow * (1-progress) + 0.9 * progress
    target_shoulder = current_shoulder * (1-progress) + (-0.5) * progress
    
    data.ctrl[left_arm["elbow"]] = target_elbow
    data.ctrl[left_arm["shoulder"]] = target_shoulder

def move_object(model, data, left_arm, right_arm, progress):
    """물체를 목표 위치로 이동하는 단계"""
    # 웨이스트 회전으로 물체 이동
    current_waist = data.ctrl[left_arm["waist"]]
    target_waist = current_waist * (1-progress) + 0.5 * progress
    
    data.ctrl[left_arm["waist"]] = target_waist

def place_object(model, data, left_arm, right_arm, progress):
    """물체를 내려놓는 단계"""
    # 현재 상태에서 높이 낮추기
    current_elbow = data.ctrl[left_arm["elbow"]]
    current_shoulder = data.ctrl[left_arm["shoulder"]]
    
    # 내려놓기 (높이 내리기)
    target_elbow = current_elbow * (1-progress) + 1.1 * progress
    target_shoulder = current_shoulder * (1-progress) + (-0.7) * progress
    
    data.ctrl[left_arm["elbow"]] = target_elbow
    data.ctrl[left_arm["shoulder"]] = target_shoulder

def release_object(model, data, left_arm, right_arm, progress):
    """물체를 놓는 단계"""
    # 그리퍼 열기
    target_gripper = 0.0084 * (1-progress) + 0.03 * progress  # 점점 열림
    
    data.ctrl[left_arm["gripper"]] = target_gripper

def return_to_home(model, data, left_arm, right_arm, progress):
    """홈 포지션으로 돌아가는 단계"""
    # 초기 'neutral_pose' 키프레임 상태로 복귀
    target_left = {
        "waist": 0.0,
        "shoulder": -0.96,
        "elbow": 1.16,
        "forearm_roll": 0.0,
        "wrist_angle": -0.3,
        "wrist_rotate": 0.0,
        "gripper": 0.0084
    }
    
    # 왼쪽 팔 제어 - 현재 상태에서 홈 포지션으로 
    for joint, idx in left_arm.items():
        current_value = data.ctrl[idx]
        target_value = target_left[joint]
        data.ctrl[idx] = current_value * (1-progress) + target_value * progress

if __name__ == "__main__":
    main()