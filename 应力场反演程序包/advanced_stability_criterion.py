# 新增的函数文件：advanced_stability_criterion.py

import numpy as np

def calculate_slip_deviation(tau, n, u):
    """
    计算滑动方向与应力场预期最大剪应力方向的偏差角
    
    输入:
    tau - 应力张量
    n - 断层面法向量
    u - 滑动方向向量
    
    输出:
    deviation - 偏差角（度）
    """
    # 计算作用在断层面上的应力向量
    traction = np.dot(tau, n)
    
    # 计算法向应力
    tau_normal = np.dot(traction, n)
    
    # 计算剪应力向量（应力向量在断层面上的投影）
    tau_shear_vector = traction - tau_normal * n
    
    # 剪应力大小
    tau_shear = np.linalg.norm(tau_shear_vector)
    
    # 避免除零错误
    if tau_shear < 1e-10:
        return 90.0
    
    # 理论滑动方向（剪应力方向）
    theoretical_slip = tau_shear_vector / tau_shear
    
    # 计算偏差角
    cos_angle = np.dot(u, theoretical_slip)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
    deviation = np.arccos(np.abs(cos_angle)) * 180.0 / np.pi
    
    return deviation


def advanced_stability_criterion(tau, friction, strike1, dip1, rake1, strike2, dip2, rake2, 
                               instability_ratio_threshold=1.4, 
                               deviation_angle_threshold_good=20.0, 
                               deviation_angle_threshold_bad=30.0):
    """
    使用两阶段策略的高级稳定性准则
    
    第一阶段：基于不稳定性比值选择
    第二阶段：基于滑动方向偏差角选择
    
    返回:
    strike, dip, rake - 选择的断层面参数
    instability - 不稳定性值
    selected_indices - 保留的数据索引
    selection_method - 每个数据的选择方法 (1: 不稳定性比值, 2: 偏差角, 0: 舍弃)
    """
    import numpy as np
    
    # 首先计算原始的不稳定性
    # 主应力和形状比
    sigma = np.sort(np.linalg.eigvals(tau))
    shape_ratio = (sigma[0]-sigma[1])/(sigma[0]-sigma[2])
    
    # 主应力方向
    diag_tensor, vector = np.linalg.eig(tau)
    value = np.linalg.eigvals(np.diag(diag_tensor))
    j = np.argsort(value)
    
    sigma_vector_1 = np.array(vector[:,j[0]])
    sigma_vector_2 = np.array(vector[:,j[1]])
    sigma_vector_3 = np.array(vector[:,j[2]])
    
    # 计算两个节面的法向量
    n1_1 = -np.sin(dip1*np.pi/180)*np.sin(strike1*np.pi/180)
    n1_2 =  np.sin(dip1*np.pi/180)*np.cos(strike1*np.pi/180)
    n1_3 = -np.cos(dip1*np.pi/180)
    
    n2_1 = -np.sin(dip2*np.pi/180)*np.sin(strike2*np.pi/180)
    n2_2 =  np.sin(dip2*np.pi/180)*np.cos(strike2*np.pi/180)
    n2_3 = -np.cos(dip2*np.pi/180)
    
    # 滑动方向
    u1_1 =  np.cos(rake1*np.pi/180)*np.cos(strike1*np.pi/180) + np.cos(dip1*np.pi/180)*np.sin(rake1*np.pi/180)*np.sin(strike1*np.pi/180)
    u1_2 =  np.cos(rake1*np.pi/180)*np.sin(strike1*np.pi/180) - np.cos(dip1*np.pi/180)*np.sin(rake1*np.pi/180)*np.cos(strike1*np.pi/180)
    u1_3 = -np.sin(rake1*np.pi/180)*np.sin(dip1*np.pi/180)
    
    u2_1 =  np.cos(rake2*np.pi/180)*np.cos(strike2*np.pi/180) + np.cos(dip2*np.pi/180)*np.sin(rake2*np.pi/180)*np.sin(strike2*np.pi/180)
    u2_2 =  np.cos(rake2*np.pi/180)*np.sin(strike2*np.pi/180) - np.cos(dip2*np.pi/180)*np.sin(rake2*np.pi/180)*np.cos(strike2*np.pi/180)
    u2_3 = -np.sin(rake2*np.pi/180)*np.sin(dip2*np.pi/180)
    
    # 在主应力坐标系中的法向量
    n1_1_ = n1_1*sigma_vector_1[0] + n1_2*sigma_vector_1[1] + n1_3*sigma_vector_1[2]
    n1_2_ = n1_1*sigma_vector_2[0] + n1_2*sigma_vector_2[1] + n1_3*sigma_vector_2[2]
    n1_3_ = n1_1*sigma_vector_3[0] + n1_2*sigma_vector_3[1] + n1_3*sigma_vector_3[2]
    
    n2_1_ = n2_1*sigma_vector_1[0] + n2_2*sigma_vector_1[1] + n2_3*sigma_vector_1[2]
    n2_2_ = n2_1*sigma_vector_2[0] + n2_2*sigma_vector_2[1] + n2_3*sigma_vector_2[2]
    n2_3_ = n2_1*sigma_vector_3[0] + n2_2*sigma_vector_3[1] + n2_3*sigma_vector_3[2]
    
    # 计算不稳定性
    tau_shear_n1_norm = np.sqrt(n1_1_**2+(1-2*shape_ratio)**2*n1_2_**2+n1_3_**2-(n1_1_**2+(1-2*shape_ratio)*n1_2_**2-n1_3_**2)**2)
    tau_normal_n1_norm = (n1_1_**2+(1-2*shape_ratio)*n1_2_**2-n1_3_**2)
    
    tau_shear_n2_norm = np.sqrt(n2_1_**2+(1-2*shape_ratio)**2*n2_2_**2+n2_3_**2-(n2_1_**2+(1-2*shape_ratio)*n2_2_**2-n2_3_**2)**2)
    tau_normal_n2_norm = (n2_1_**2+(1-2*shape_ratio)*n2_2_**2-n2_3_**2)
    
    instability_n1 = (tau_shear_n1_norm - friction*(tau_normal_n1_norm-1))/(friction+np.sqrt(1+friction**2))
    instability_n2 = (tau_shear_n2_norm - friction*(tau_normal_n2_norm-1))/(friction+np.sqrt(1+friction**2))
    
    # 初始化输出数组
    N = np.size(strike1)
    strike = np.zeros(N)
    dip = np.zeros(N)
    rake = np.zeros(N)
    instability = np.zeros(N)
    selected_indices = []
    selection_method = np.zeros(N)
    
    # 对每个震源机制进行处理
    for i in range(N):
        # 第一阶段：基于不稳定性比值
        if instability_n1[i] > 0 and instability_n2[i] > 0:
            ratio = max(instability_n1[i], instability_n2[i]) / min(instability_n1[i], instability_n2[i])
            
            if ratio >= instability_ratio_threshold:
                # 选择更不稳定的面
                if instability_n1[i] >= instability_n2[i]:
                    strike[i] = strike1[i]
                    dip[i] = dip1[i]
                    rake[i] = rake1[i]
                    instability[i] = instability_n1[i]
                else:
                    strike[i] = strike2[i]
                    dip[i] = dip2[i]
                    rake[i] = rake2[i]
                    instability[i] = instability_n2[i]
                selected_indices.append(i)
                selection_method[i] = 1
                continue
        
        # 第二阶段：基于滑动方向偏差角
        # 构建法向量和滑动向量
        n1 = np.array([n1_1[i], n1_2[i], n1_3[i]])
        n2 = np.array([n2_1[i], n2_2[i], n2_3[i]])
        u1 = np.array([u1_1[i], u1_2[i], u1_3[i]])
        u2 = np.array([u2_1[i], u2_2[i], u2_3[i]])
        
        # 计算偏差角
        deviation1 = calculate_slip_deviation(tau, n1, u1)
        deviation2 = calculate_slip_deviation(tau, n2, u2)
        
        # 判断选择
        plane1_good = deviation1 < deviation_angle_threshold_good
        plane2_good = deviation2 < deviation_angle_threshold_good
        plane1_bad = deviation1 > deviation_angle_threshold_bad
        plane2_bad = deviation2 > deviation_angle_threshold_bad
        
        if plane1_good and plane2_bad:
            # 选择节面1
            strike[i] = strike1[i]
            dip[i] = dip1[i]
            rake[i] = rake1[i]
            instability[i] = instability_n1[i]
            selected_indices.append(i)
            selection_method[i] = 2
        elif plane2_good and plane1_bad:
            # 选择节面2
            strike[i] = strike2[i]
            dip[i] = dip2[i]
            rake[i] = rake2[i]
            instability[i] = instability_n2[i]
            selected_indices.append(i)
            selection_method[i] = 2
        else:
            # 舍弃该数据
            selection_method[i] = 0
    
    # 只返回被选择的数据
    selected_indices = np.array(selected_indices)
    if len(selected_indices) > 0:
        strike = strike[selected_indices]
        dip = dip[selected_indices]
        rake = rake[selected_indices]
        instability = instability[selected_indices]
    else:
        strike = np.array([])
        dip = np.array([])
        rake = np.array([])
        instability = np.array([])
    
    return strike, dip, rake, instability, selected_indices, selection_method