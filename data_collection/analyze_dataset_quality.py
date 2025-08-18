#!/usr/bin/env python3
"""
강화학습 데이터셋 품질 분석 도구
- VR → Joint 매핑 정확도 검증
- End-effector 일관성 분석
- 강화학습 적합성 평가
"""

import json
import numpy as np
import mujoco

def load_dataset(file_path):
    """데이터셋 로드"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_vr_joint_consistency(samples):
    """VR Pose → Joint 매핑 일관성 분석"""
    print("🔍 VR Pose → Joint 매핑 일관성 분석")
    
    vr_positions = []
    vr_orientations = []
    joint_angles = []
    ee_positions = []
    
    for sample in samples:
        vr_positions.append(sample['vr_position_delta'])
        vr_orientations.append(sample['vr_orientation_delta']) 
        joint_angles.append(sample['joint_angles'])
        ee_positions.append(sample['end_effector_position'])
    
    vr_positions = np.array(vr_positions)
    vr_orientations = np.array(vr_orientations)
    joint_angles = np.array(joint_angles)
    ee_positions = np.array(ee_positions)
    
    # 1. VR Position 변화량 vs Joint 변화량 상관관계
    print(f"📊 데이터 포인트: {len(samples)}개")
    print(f"📍 VR 위치 범위: X[{vr_positions[:,0].min():.3f}, {vr_positions[:,0].max():.3f}]")
    print(f"                Y[{vr_positions[:,1].min():.3f}, {vr_positions[:,1].max():.3f}]") 
    print(f"                Z[{vr_positions[:,2].min():.3f}, {vr_positions[:,2].max():.3f}]")
    
    # 2. Joint 각도 변화의 부드러움
    joint_velocities = np.diff(joint_angles, axis=0)
    joint_accelerations = np.diff(joint_velocities, axis=0)
    
    print(f"🤖 Joint 각속도 표준편차: {np.std(joint_velocities, axis=0)}")
    print(f"🤖 Joint 각가속도 표준편차: {np.std(joint_accelerations, axis=0)}")
    
    # 3. End-effector 위치 일관성
    ee_velocities = np.diff(ee_positions, axis=0)
    ee_velocity_norms = np.linalg.norm(ee_velocities, axis=1)
    
    print(f"🎯 End-effector 속도 평균: {np.mean(ee_velocity_norms):.6f}m/step")
    print(f"🎯 End-effector 속도 표준편차: {np.std(ee_velocity_norms):.6f}")
    
    return {
        'vr_positions': vr_positions,
        'joint_angles': joint_angles, 
        'ee_positions': ee_positions,
        'joint_smoothness': np.std(joint_velocities, axis=0),
        'ee_smoothness': np.std(ee_velocity_norms)
    }

def check_kinematic_accuracy(samples):
    """운동학적 정확도 검증"""
    print("\n🎯 운동학적 정확도 검증")
    
    # MuJoCo 모델 로드 (Forward Kinematics 검증용)
    try:
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)
        
        actual_ee_positions = []
        calculated_ee_positions = []
        
        for sample in samples[:10]:  # 처음 10개 샘플만 검증
            # 관절 각도 설정
            joint_angles = sample['joint_angles']
            data.qpos[:4] = joint_angles
            
            # Forward kinematics 계산
            mujoco.mj_forward(model, data)
            
            # End-effector 위치 계산 (link4 끝점)
            ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_link')
            if ee_site_id >= 0:
                calculated_pos = data.site_xpos[ee_site_id].copy()
                actual_pos = sample['end_effector_position']
                
                actual_ee_positions.append(actual_pos)
                calculated_ee_positions.append(calculated_pos)
        
        if actual_ee_positions:
            actual_ee = np.array(actual_ee_positions)
            calculated_ee = np.array(calculated_ee_positions)
            
            # 위치 오차 계산
            position_errors = np.linalg.norm(actual_ee - calculated_ee, axis=1)
            avg_error = np.mean(position_errors)
            max_error = np.max(position_errors)
            
            print(f"📏 평균 위치 오차: {avg_error:.6f}m")
            print(f"📏 최대 위치 오차: {max_error:.6f}m")
            print(f"📏 정확도 점수: {100*(1-avg_error):.1f}%")
            
            return avg_error < 0.01  # 1cm 이하면 양호
        
    except Exception as e:
        print(f"⚠️ 운동학 검증 실패: {e}")
        return None

def evaluate_rl_suitability(analysis_results):
    """강화학습 적합성 평가"""
    print("\n🤖 강화학습 데이터셋 적합성 평가")
    
    # 1. 데이터 다양성
    joint_ranges = np.ptp(analysis_results['joint_angles'], axis=0)
    ee_range = np.ptp(analysis_results['ee_positions'], axis=0)
    
    print(f"📊 Joint 움직임 범위: {joint_ranges}")
    print(f"📊 End-effector 작업공간: X={ee_range[0]:.3f}m, Y={ee_range[1]:.3f}m, Z={ee_range[2]:.3f}m")
    
    # 2. 부드러움 점수
    smoothness_score = 1.0 / (1.0 + np.mean(analysis_results['joint_smoothness']))
    ee_smoothness_score = 1.0 / (1.0 + analysis_results['ee_smoothness'])
    
    print(f"🌊 Joint 부드러움 점수: {smoothness_score:.3f}/1.0")
    print(f"🌊 End-effector 부드러움 점수: {ee_smoothness_score:.3f}/1.0")
    
    # 3. 종합 평가
    diversity_score = min(np.sum(joint_ranges > 0.5) / 4.0, 1.0)  # 4개 관절 중 절반 이상 움직임
    workspace_score = min(np.sum(ee_range > 0.1) / 3.0, 1.0)     # 3축 모두 10cm 이상 움직임
    
    overall_score = (diversity_score + smoothness_score + ee_smoothness_score + workspace_score) / 4.0
    
    print(f"\n📈 종합 평가:")
    print(f"   다양성: {diversity_score:.2f}/1.0")
    print(f"   부드러움: {(smoothness_score + ee_smoothness_score)/2:.2f}/1.0") 
    print(f"   작업공간: {workspace_score:.2f}/1.0")
    print(f"   종합 점수: {overall_score:.2f}/1.0")
    
    # 강화학습 적합성 판정
    if overall_score >= 0.7:
        suitability = "✅ 우수 - 강화학습에 적합"
    elif overall_score >= 0.5:
        suitability = "⚠️ 보통 - 개선 필요하지만 사용 가능"
    else:
        suitability = "❌ 부족 - 데이터 품질 개선 필요"
    
    print(f"\n🎯 강화학습 적합성: {suitability}")
    
    return {
        'overall_score': overall_score,
        'diversity_score': diversity_score,
        'smoothness_score': (smoothness_score + ee_smoothness_score)/2,
        'workspace_score': workspace_score,
        'suitability': suitability
    }

def suggest_improvements(analysis_results, rl_evaluation):
    """개선 방안 제시"""
    print("\n💡 개선 방안 제시")
    
    if rl_evaluation['diversity_score'] < 0.6:
        print("📊 데이터 다양성 부족:")
        print("   - 더 넓은 작업공간에서 데이터 수집")
        print("   - 다양한 관절 구성으로 같은 위치 도달")
        print("   - 복잡한 궤적 추가 (곡선, 원형 등)")
    
    if rl_evaluation['smoothness_score'] < 0.6:
        print("🌊 움직임 부드러움 부족:")
        print("   - VR 컨트롤러 더 천천히 움직이기")
        print("   - 필터링 강화 (test3.py의 smoothing_factor 증가)")
        print("   - 급격한 방향 전환 피하기")
    
    if rl_evaluation['workspace_score'] < 0.6:
        print("🎯 작업공간 활용 부족:")
        print("   - 로봇 도달 가능 영역 전체 활용")
        print("   - 높이 변화 더 많이 포함")
        print("   - 좌우 움직임 확대")
    
    print("\n🚀 강화학습 최적화를 위한 추가 제안:")
    print("   - 성공/실패 라벨 추가")
    print("   - 중간 목표점(waypoint) 기록")
    print("   - 그리퍼 상태와 객체 상호작용 정보")
    print("   - 시간 정보 정규화")

if __name__ == "__main__":
    print("🤖 강화학습 데이터셋 품질 분석 시작\n")
    
    # 가장 큰 데이터셋 분석
    file_path = 'keyboard_collected_mapping_20250808_232704.json'
    
    try:
        dataset = load_dataset(file_path)
        samples = dataset['samples']
        
        print(f"📁 분석 파일: {file_path}")
        print(f"📊 총 샘플 수: {len(samples)}개\n")
        
        # 1. VR-Joint 매핑 분석
        analysis_results = analyze_vr_joint_consistency(samples)
        
        # 2. 운동학적 정확도 검증
        kinematic_ok = check_kinematic_accuracy(samples)
        
        # 3. 강화학습 적합성 평가
        rl_evaluation = evaluate_rl_suitability(analysis_results)
        
        # 4. 개선 방안 제시
        suggest_improvements(analysis_results, rl_evaluation)
        
        print(f"\n{'='*50}")
        print(f"🎯 최종 결론: {rl_evaluation['suitability']}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")