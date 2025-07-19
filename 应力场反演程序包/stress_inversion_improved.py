# stress_inversion_improved.py
# 改进版的应力反演函数，使用新的节面选择策略

import numpy as np
from advanced_stability_criterion import advanced_stability_criterion

def linear_stress_inversion_Michael(strike1,dip1,rake1,strike2,dip2,rake2):
    """
    使用Michael方法的线性应力反演（随机选择断层面）
    """
    N = np.size(strike1)
    
    strike = np.zeros((N))
    dip = np.zeros((N))
    rake = np.zeros((N))
    
    # 随机选择断层面
    for i_mechanism in range(N):
        choice = np.random.randint(2)
        
        if choice == 1:
            strike[i_mechanism] = strike1[i_mechanism]
            dip[i_mechanism] = dip1[i_mechanism]
            rake[i_mechanism] = rake1[i_mechanism]
        else:
            strike[i_mechanism] = strike2[i_mechanism]
            dip[i_mechanism] = dip2[i_mechanism]
            rake[i_mechanism] = rake2[i_mechanism]
    
    # 断层法向量和滑动方向
    u1 =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u2 =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u3 = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
    
    n1 = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2 =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3 = -np.cos(dip*np.pi/180)
    
    # 构建系数矩阵
    A11_n =  n1*(1-n1**2)
    A21_n = -n1*n2**2
    A31_n = -n1*n3**2
    A41_n = -2*n1*n2*n3
    A51_n =  n3*(1-2*n1**2)
    A61_n =  n2*(1-2*n1**2)
    
    A12_n = -n2*n1**2
    A22_n =  n2*(1-n2**2)
    A32_n = -n2*n3**2
    A42_n =  n3*(1-2*n2**2)
    A52_n = -2*n1*n2*n3
    A62_n =  n1*(1-2*n2**2)
    
    A13_n = -n3*n1**2
    A23_n = -n3*n2**2
    A33_n =  n3*(1-n3**2)
    A43_n =  n2*(1-2*n3**2)
    A53_n =  n1*(1-2*n3**2)
    A63_n = -2*n1*n2*n3
    
    A1 = np.transpose([A11_n, A21_n, A31_n, A41_n, A51_n, A61_n])
    A2 = np.transpose([A12_n, A22_n, A32_n, A42_n, A52_n, A62_n])
    A3 = np.transpose([A13_n, A23_n, A33_n, A43_n, A53_n, A63_n])
    
    a_vector_1 = u1
    a_vector_2 = u2
    a_vector_3 = u3
    
    A = np.r_[A1, A2, A3]
    a_vector = np.r_[a_vector_1, a_vector_2, a_vector_3]
    
    # 零迹条件
    A = np.append(A, [[1., 1., 1., 0, 0, 0]], axis=0)
    a_vector = np.append(a_vector, [0])
    
    # 广义逆求解
    stress_vector = np.real(np.dot(np.linalg.pinv(A), a_vector))
    stress_tensor = np.array([[stress_vector[0], stress_vector[5], stress_vector[4]],
                              [stress_vector[5], stress_vector[1], stress_vector[3]],
                              [stress_vector[4], stress_vector[3], stress_vector[2]]])
    
    sigma = np.linalg.eigvals(stress_tensor)
    stress = stress_tensor/max(abs(sigma))
    
    return stress

def linear_stress_inversion(strike, dip, rake):
    """
    从给定的断层面参数进行线性应力反演
    """
    N = np.size(strike)
    
    # 断层法向量和滑动方向
    u1 =  np.cos(rake*np.pi/180)*np.cos(strike*np.pi/180) + np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.sin(strike*np.pi/180)
    u2 =  np.cos(rake*np.pi/180)*np.sin(strike*np.pi/180) - np.cos(dip*np.pi/180)*np.sin(rake*np.pi/180)*np.cos(strike*np.pi/180)
    u3 = -np.sin(rake*np.pi/180)*np.sin(dip*np.pi/180)
    
    n1 = -np.sin(dip*np.pi/180)*np.sin(strike*np.pi/180)
    n2 =  np.sin(dip*np.pi/180)*np.cos(strike*np.pi/180)
    n3 = -np.cos(dip*np.pi/180)
    
    # 构建系数矩阵（与上面相同的代码）
    A11_n =  n1*(1-n1**2)
    A21_n = -n1*n2**2
    A31_n = -n1*n3**2
    A41_n = -2*n1*n2*n3
    A51_n =  n3*(1-2*n1**2)
    A61_n =  n2*(1-2*n1**2)
    
    A12_n = -n2*n1**2
    A22_n =  n2*(1-n2**2)
    A32_n = -n2*n3**2
    A42_n =  n3*(1-2*n2**2)
    A52_n = -2*n1*n2*n3
    A62_n =  n1*(1-2*n2**2)
    
    A13_n = -n3*n1**2
    A23_n = -n3*n2**2
    A33_n =  n3*(1-n3**2)
    A43_n =  n2*(1-2*n3**2)
    A53_n =  n1*(1-2*n3**2)
    A63_n = -2*n1*n2*n3
    
    A1 = np.transpose([A11_n, A21_n, A31_n, A41_n, A51_n, A61_n])
    A2 = np.transpose([A12_n, A22_n, A32_n, A42_n, A52_n, A62_n])
    A3 = np.transpose([A13_n, A23_n, A33_n, A43_n, A53_n, A63_n])
    
    a_vector_1 = u1
    a_vector_2 = u2
    a_vector_3 = u3
    
    A = np.r_[A1, A2, A3]
    a_vector = np.r_[a_vector_1, a_vector_2, a_vector_3]
    
    # 零迹条件
    A = np.append(A, [[1., 1., 1., 0, 0, 0]], axis=0)
    a_vector = np.append(a_vector, [0])
    
    # 广义逆求解
    stress_vector = np.real(np.dot(np.linalg.pinv(A), a_vector))
    stress_tensor = np.array([[stress_vector[0], stress_vector[5], stress_vector[4]],
                              [stress_vector[5], stress_vector[1], stress_vector[3]],
                              [stress_vector[4], stress_vector[3], stress_vector[2]]])
    
    sigma = np.linalg.eigvals(stress_tensor)
    stress = stress_tensor/max(abs(sigma))
    
    return stress

def stress_inversion_improved(strike1_orig, dip1_orig, rake1_orig, 
                            strike2_orig, dip2_orig, rake2_orig,
                            friction_min, friction_max, friction_step,
                            N_iterations, N_realizations,
                            instability_ratio_threshold=1.4,
                            deviation_angle_good=20.0,
                            deviation_angle_bad=30.0):
    """
    改进的应力反演函数，使用两阶段节面选择策略
    
    返回:
    tau_optimum - 最优应力张量
    shape_ratio - 应力形状比
    strike, dip, rake - 选择的断层面参数
    instability - 不稳定性值
    friction_optimum - 最优摩擦系数
    selected_indices - 最终使用的数据索引
    selection_stats - 选择统计信息
    """
    
    # 初始应力估计（使用Michael方法）
    tau = np.zeros((3,3))
    for i_realization in range(N_realizations):
        tau_realization = linear_stress_inversion_Michael(strike1_orig, dip1_orig, rake1_orig,
                                                         strike2_orig, dip2_orig, rake2_orig)
        tau = tau + tau_realization
    
    tau0 = tau/np.linalg.norm(tau, 2)
    
    # 搜索最优摩擦系数
    friction_range = np.arange(friction_min, friction_max, friction_step)
    mean_instability = np.zeros(len(friction_range))
    n_selected = np.zeros(len(friction_range))
    
    for i_friction, friction in enumerate(friction_range):
        # 迭代优化
        for i_iteration in range(N_iterations):
            # 使用改进的稳定性准则
            strike, dip, rake, instability, selected_indices, selection_method = \
                advanced_stability_criterion(tau0, friction, 
                                           strike1_orig, dip1_orig, rake1_orig,
                                           strike2_orig, dip2_orig, rake2_orig,
                                           instability_ratio_threshold,
                                           deviation_angle_good,
                                           deviation_angle_bad)
            
            if len(strike) > 0:
                tau = linear_stress_inversion(strike, dip, rake)
                tau0 = tau
            else:
                # 如果没有数据被选择，保持原应力
                pass
        
        if len(instability) > 0:
            mean_instability[i_friction] = np.mean(instability)
            n_selected[i_friction] = len(instability)
        else:
            mean_instability[i_friction] = 0
            n_selected[i_friction] = 0
    
    # 选择最优摩擦系数（考虑选择的数据量）
    # 只考虑至少保留了50%数据的摩擦系数
    min_data_fraction = 0.5
    min_data_count = int(len(strike1_orig) * min_data_fraction)
    
    valid_frictions = n_selected >= min_data_count
    if np.any(valid_frictions):
        valid_mean_instability = mean_instability * valid_frictions
        i_optimum = np.argmax(valid_mean_instability)
    else:
        # 如果没有摩擦系数能保留足够的数据，选择保留数据最多的
        i_optimum = np.argmax(n_selected)
    
    friction_optimum = friction_range[i_optimum]
    
    # 使用最优摩擦系数进行最终反演
    tau0 = tau/np.linalg.norm(tau, 2)  # 重置初始应力
    for i_iteration in range(N_iterations):
        strike, dip, rake, instability, selected_indices, selection_method = \
            advanced_stability_criterion(tau0, friction_optimum,
                                       strike1_orig, dip1_orig, rake1_orig,
                                       strike2_orig, dip2_orig, rake2_orig,
                                       instability_ratio_threshold,
                                       deviation_angle_good,
                                       deviation_angle_bad)
        
        if len(strike) > 0:
            tau = linear_stress_inversion(strike, dip, rake)
            tau0 = tau
    
    # 计算统计信息
    n_total = len(strike1_orig)
    n_selected_instability = np.sum(selection_method == 1)
    n_selected_deviation = np.sum(selection_method == 2)
    n_discarded = np.sum(selection_method == 0)
    
    selection_stats = {
        'n_total': n_total,
        'n_selected': len(selected_indices),
        'n_selected_instability': n_selected_instability,
        'n_selected_deviation': n_selected_deviation,
        'n_discarded': n_discarded,
        'selection_method': selection_method
    }
    
    # 计算应力形状比
    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])
    
    return tau, shape_ratio, strike, dip, rake, instability, friction_optimum, selected_indices, selection_stats