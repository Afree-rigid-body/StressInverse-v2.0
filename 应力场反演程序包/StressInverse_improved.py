#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进版应力反演程序
增加了两阶段节面选择策略和交互式输入
"""

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 导入改进的模块
sys.path.append(os.path.dirname(__file__))

# 读取输入参数（GUI版本）
try:
    import GUI_Input_parameters as ip
except ImportError:
    # 如果GUI版本失败，使用简单的文件对话框版本
    import Input_parameters_interactive as ip

# 导入其他必要的模块
import read_mechanism as rm
from stress_inversion_improved import stress_inversion_improved, linear_stress_inversion
from advanced_stability_criterion import advanced_stability_criterion
import azimuth_plunge as ap
import slip_deviation as sd
import principal_mechanisms as pm
import noisy_mechanisms as nm
from statistics_stress_inversion import statistics_stress_inversion
import plot_stress as plots
import plot_mohr as plotm
import plot_stress_axes as plotsa
import scipy.io as sio

print("\n" + "="*60)
print("改进版应力反演程序")
print("="*60)

# 读取震源机制数据
print("\n读取震源机制数据...")
strike_orig_1, dip_orig_1, rake_orig_1, strike_orig_2, dip_orig_2, rake_orig_2 = rm.read_mechanisms(ip.input_file)
print(f"读取到 {len(strike_orig_1)} 个震源机制")

# 应力反演（使用改进的方法）
print("\n开始应力反演...")
tau_optimum, shape_ratio, strike, dip, rake, instability, friction, selected_indices, selection_stats = \
    stress_inversion_improved(strike_orig_1, dip_orig_1, rake_orig_1,
                            strike_orig_2, dip_orig_2, rake_orig_2,
                            ip.friction_min, ip.friction_max, ip.friction_step,
                            ip.N_iterations, ip.N_realizations,
                            ip.instability_ratio_threshold,
                            ip.deviation_angle_good,
                            ip.deviation_angle_bad)

# 打印选择统计信息
print("\n节面选择统计:")
print(f"- 总震源机制数: {selection_stats['n_total']}")
print(f"- 通过不稳定性比值选择: {selection_stats['n_selected_instability']}")
print(f"- 通过滑动方向偏差角选择: {selection_stats['n_selected_deviation']}")
print(f"- 被舍弃的数据: {selection_stats['n_discarded']}")
print(f"- 最终使用的数据: {selection_stats['n_selected']} ({selection_stats['n_selected']/selection_stats['n_total']*100:.1f}%)")
print(f"\n最优摩擦系数: {friction:.2f}")
print(f"应力形状比 R: {shape_ratio:.3f}")

# 计算主应力轴方向
diag_tensor, vector = np.linalg.eig(tau_optimum)
value = np.linalg.eigvals(np.diag(diag_tensor))
j = np.argsort(value)

sigma_vector_1_optimum = np.array(vector[:,j[0]])
sigma_vector_2_optimum = np.array(vector[:,j[1]])
sigma_vector_3_optimum = np.array(vector[:,j[2]])

direction_sigma_1, direction_sigma_2, direction_sigma_3 = ap.azimuth_plunge(tau_optimum)

print(f"\n主应力轴方向:")
print(f"σ1: 方位角 {direction_sigma_1[0]:.1f}°, 倾伏角 {direction_sigma_1[1]:.1f}°")
print(f"σ2: 方位角 {direction_sigma_2[0]:.1f}°, 倾伏角 {direction_sigma_2[1]:.1f}°")
print(f"σ3: 方位角 {direction_sigma_3[0]:.1f}°, 倾伏角 {direction_sigma_3[1]:.1f}°")

# 计算滑动偏差
if len(strike) > 0:
    slip_deviation_1, slip_deviation_2 = sd.slip_deviation(tau_optimum, strike, dip, rake)
    print(f"\n平均滑动偏差: {np.mean(np.minimum(slip_deviation_1, slip_deviation_2)):.1f}°")

# 计算主震源机制
principal_strike, principal_dip, principal_rake = pm.principal_mechanisms(
    sigma_vector_1_optimum, sigma_vector_3_optimum, friction)

print(f"\n主震源机制:")
print(f"断层面1: 走向 {principal_strike[0]:.1f}°, 倾角 {principal_dip[0]:.1f}°, 滑动角 {principal_rake[0]:.1f}°")
print(f"断层面2: 走向 {principal_strike[1]:.1f}°, 倾角 {principal_dip[1]:.1f}°, 滑动角 {principal_rake[1]:.1f}°")

# 噪声分析（仅对被选择的数据进行）
print(f"\n进行不确定性分析 ({ip.N_noise_realizations} 次噪声实现)...")

# 只使用被选择的原始数据进行噪声分析
strike1_selected = strike_orig_1[selected_indices]
dip1_selected = dip_orig_1[selected_indices]
rake1_selected = rake_orig_1[selected_indices]
strike2_selected = strike_orig_2[selected_indices]
dip2_selected = dip_orig_2[selected_indices]
rake2_selected = rake_orig_2[selected_indices]

sigma_vector_1_statistics = np.zeros((3, ip.N_noise_realizations))
sigma_vector_2_statistics = np.zeros((3, ip.N_noise_realizations))
sigma_vector_3_statistics = np.zeros((3, ip.N_noise_realizations))
shape_ratio_statistics = np.zeros(ip.N_noise_realizations)

for i in range(ip.N_noise_realizations):
    # 对选择的数据添加噪声
    strike1_noisy, dip1_noisy, rake1_noisy, _, _, _, n_error, u_error = \
        nm.noisy_mechanisms(ip.mean_deviation, strike1_selected, dip1_selected, rake1_selected)
    
    # 使用带噪声的数据进行反演（这里仍使用原始的稳定性准则）
    sigma_vector_1, sigma_vector_2, sigma_vector_3, shape_ratio_noisy = \
        statistics_stress_inversion(strike1_noisy, dip1_noisy, rake1_noisy,
                                   strike2_selected, dip2_selected, rake2_selected,
                                   friction, ip.N_iterations, ip.N_realizations)
    
    sigma_vector_1_statistics[:, i] = sigma_vector_1
    sigma_vector_2_statistics[:, i] = sigma_vector_2
    sigma_vector_3_statistics[:, i] = sigma_vector_3
    shape_ratio_statistics[i] = shape_ratio_noisy

# 计算误差统计
sigma_1_error_statistics = np.zeros(ip.N_noise_realizations)
sigma_2_error_statistics = np.zeros(ip.N_noise_realizations)
sigma_3_error_statistics = np.zeros(ip.N_noise_realizations)

for i in range(ip.N_noise_realizations):
    sigma_1_error_statistics[i] = np.arccos(np.abs(np.dot(sigma_vector_1_statistics[:,i], sigma_vector_1_optimum)))*180/np.pi
    sigma_2_error_statistics[i] = np.arccos(np.abs(np.dot(sigma_vector_2_statistics[:,i], sigma_vector_2_optimum)))*180/np.pi
    sigma_3_error_statistics[i] = np.arccos(np.abs(np.dot(sigma_vector_3_statistics[:,i], sigma_vector_3_optimum)))*180/np.pi

print(f"\n不确定性分析结果:")
print(f"σ1方向最大误差: {np.max(sigma_1_error_statistics):.1f}°")
print(f"σ2方向最大误差: {np.max(sigma_2_error_statistics):.1f}°")
print(f"σ3方向最大误差: {np.max(sigma_3_error_statistics):.1f}°")
print(f"形状比R标准差: {np.std(shape_ratio_statistics):.3f}")

# 保存结果
print("\n保存结果...")

# 准备输出数据
mechanisms_data = np.column_stack([strike, dip, rake])
principal_mechanisms_data = np.column_stack([principal_strike, principal_dip, principal_rake])

# 保存选择信息
selection_info = np.column_stack([
    np.arange(len(strike_orig_1)),
    selection_stats['selection_method'],
    strike_orig_1, dip_orig_1, rake_orig_1,
    strike_orig_2, dip_orig_2, rake_orig_2
])

# 保存文本文件
np.savetxt(ip.output_file + "_mechanisms.dat", mechanisms_data, 
           fmt='%10.4f %10.4f %10.4f',
           header='strike     dip        rake')
np.savetxt(ip.principal_mechanisms_file + ".dat", principal_mechanisms_data,
           fmt='%10.4f %10.4f %10.4f',
           header='strike     dip        rake')
np.savetxt(ip.output_file + "_selection_info.dat", selection_info,
           fmt='%5d %5d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f',
           header='index method strike1 dip1 rake1 strike2 dip2 rake2 (method: 0=discarded, 1=instability, 2=deviation)')

# 保存MATLAB格式文件
data_matlab = {
    'tau_optimum': tau_optimum,
    'sigma_1': {'azimuth': direction_sigma_1[0], 'plunge': direction_sigma_1[1]},
    'sigma_2': {'azimuth': direction_sigma_2[0], 'plunge': direction_sigma_2[1]},
    'sigma_3': {'azimuth': direction_sigma_3[0], 'plunge': direction_sigma_3[1]},
    'shape_ratio': shape_ratio,
    'friction': friction,
    'mechanisms': {'strike': strike, 'dip': dip, 'rake': rake},
    'principal_mechanisms': {'strike': principal_strike, 'dip': principal_dip, 'rake': principal_rake},
    'selection_stats': selection_stats,
    'selected_indices': selected_indices
}
sio.savemat(ip.output_file + ".mat", data_matlab)

# 生成图形
print("\n生成图形...")

# P/T轴和主应力轴图
if len(strike) > 0:
    plots.plot_stress(tau_optimum, strike, dip, rake, ip.P_T_plot)
    print("- P/T轴图完成")

# 莫尔圆图
if len(strike) > 0:
    plotm.plot_mohr(tau_optimum, strike, dip, rake, principal_strike, principal_dip, principal_rake, ip.Mohr_plot)
    print("- 莫尔圆图完成")

# 主应力轴置信区间图
plotsa.plot_stress_axes(sigma_vector_1_statistics, sigma_vector_2_statistics, sigma_vector_3_statistics, ip.stress_plot)
print("- 应力轴置信区间图完成")

# 形状比直方图
fig, ax = plt.subplots(figsize=(8, 6))
n, bins, patches = ax.hist(shape_ratio_statistics, bins=ip.shape_ratio_axis, 
                          color='#0504aa', alpha=0.7, edgecolor='black')
ax.axvline(shape_ratio, color='red', linestyle='--', linewidth=2, label=f'Optimal value R={shape_ratio:.3f}')
ax.set_xlabel('Stress Shape Ratio R')
ax.set_ylabel('Frequency')
ax.set_title('Stress Shape Ratio Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(ip.shape_ratio_plot + '.png', dpi=300, bbox_inches='tight')
plt.close()
print("- 形状比直方图完成")

print("\n" + "="*60)
print("程序运行完成！")
print("="*60)