# Input_parameters_interactive.py
# 改进版的输入参数文件，支持图形界面文件选择

import numpy as np
import os
import sys

def get_input_file():
    """使用图形界面选择输入文件"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # 创建隐藏的根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 设置初始目录（如果存在）
        initial_dir = '../Data' if os.path.exists('../Data') else os.getcwd()
        
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择震源机制数据文件",
            initialdir=initial_dir,
            filetypes=[
                ("数据文件", "*.dat *.txt"),
                ("DAT文件", "*.dat"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        
        # 销毁根窗口
        root.destroy()
        
        # 检查是否选择了文件
        if not file_path:
            print("错误：未选择文件")
            return None
        
        return file_path
        
    except ImportError:
        print("警告：tkinter未安装，使用命令行输入模式")
        return get_input_file_cli()

def get_input_file_cli():
    """命令行方式输入文件路径"""
    print("\n请输入震源机制数据文件的完整路径：")
    print("（提示：可以直接拖拽文件到终端窗口）")
    
    while True:
        file_path = input("文件路径: ").strip()
        
        # 移除可能的引号
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        elif file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]
        
        if not file_path:
            print("错误：文件路径不能为空")
            continue
            
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 - {file_path}")
            continue
            
        return file_path

# 使用图形界面选择输入文件
print("正在打开文件选择窗口...")
input_file = get_input_file()
if input_file is None:
    print("程序退出：未选择输入文件")
    sys.exit(1)

print(f"\n已选择文件: {os.path.basename(input_file)}")
print(f"完整路径: {input_file}")

# 基于输入文件名生成输出文件名
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_dir = '../Output_improved'
figure_dir = '../Figures_improved'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# 输出文件
output_file = os.path.join(output_dir, f'{base_name}_output')
principal_mechanisms_file = os.path.join(output_dir, f'{base_name}_principal_mechanisms')

# 图形文件
shape_ratio_plot = os.path.join(figure_dir, f'{base_name}_shape_ratio')
stress_plot = os.path.join(figure_dir, f'{base_name}_stress_directions')
P_T_plot = os.path.join(figure_dir, f'{base_name}_P_T_axes')
Mohr_plot = os.path.join(figure_dir, f'{base_name}_Mohr_circles')
faults_plot = os.path.join(figure_dir, f'{base_name}_faults')

# 精度参数
N_noise_realizations = 100  # 噪声实现次数
mean_deviation = 5          # 震源机制误差的标准差（度）

# 高级控制参数
N_iterations = 6            # 应力反演迭代次数
N_realizations = 10         # 随机选择断层的初始应力反演次数

# 应力形状比直方图参数
shape_ratio_min = 0
shape_ratio_max = 1
shape_ratio_step = 0.025
shape_ratio_axis = np.arange(shape_ratio_min+0.0125, shape_ratio_max, shape_ratio_step)

# 摩擦系数搜索范围
friction_min = 0.40
friction_max = 1.00
friction_step = 0.05

# 两阶段节面选择策略的参数
instability_ratio_threshold = 1.4    # 不稳定性比值阈值
deviation_angle_good = 20.0          # 滑动方向偏差角的"好"阈值（度）
deviation_angle_bad = 30.0           # 滑动方向偏差角的"坏"阈值（度）

print("\n参数设置：")
print(f"- 不稳定性比值阈值: {instability_ratio_threshold}")
print(f"- 滑动方向偏差角阈值: 好 < {deviation_angle_good}°, 坏 > {deviation_angle_bad}°")
print(f"- 噪声实现次数: {N_noise_realizations}")
print(f"- 震源机制误差标准差: {mean_deviation}°")