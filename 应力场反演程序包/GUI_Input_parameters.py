# GUI_Input_parameters.py
# 带有完整图形界面的输入参数设置

import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys

class StressInversionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("应力反演程序 - 参数设置")
        self.root.geometry("600x750")
        
        # 设置变量
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="../Output_improved")
        self.figure_dir = tk.StringVar(value="../Figures_improved")
        
        # 数值参数
        self.instability_threshold = tk.DoubleVar(value=1.4)
        self.deviation_good = tk.DoubleVar(value=20.0)
        self.deviation_bad = tk.DoubleVar(value=30.0)
        self.noise_realizations = tk.IntVar(value=100)
        self.mean_deviation = tk.DoubleVar(value=5.0)
        self.n_iterations = tk.IntVar(value=6)
        self.n_realizations = tk.IntVar(value=10)
        self.friction_min = tk.DoubleVar(value=0.40)
        self.friction_max = tk.DoubleVar(value=1.00)
        self.friction_step = tk.DoubleVar(value=0.05)
        
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 文件选择部分
        file_frame = ttk.LabelFrame(main_frame, text="文件设置", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 输入文件
        ttk.Label(file_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.input_file, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_input_file).grid(row=0, column=2)
        
        # 输出目录
        ttk.Label(file_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_output_dir).grid(row=1, column=2)
        
        # 图形目录
        ttk.Label(file_frame, text="图形目录:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.figure_dir, width=40).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_figure_dir).grid(row=2, column=2)
        
        # 节面选择参数
        selection_frame = ttk.LabelFrame(main_frame, text="节面选择参数", padding="10")
        selection_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(selection_frame, text="不稳定性比值阈值:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(selection_frame, from_=1.1, to=2.0, increment=0.1, 
                    textvariable=self.instability_threshold, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(selection_frame, text="滑动方向偏差角-好 (度):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(selection_frame, from_=10, to=30, increment=5, 
                    textvariable=self.deviation_good, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(selection_frame, text="滑动方向偏差角-坏 (度):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(selection_frame, from_=20, to=45, increment=5, 
                    textvariable=self.deviation_bad, width=10).grid(row=2, column=1, sticky=tk.W)
        
        # 不确定性分析参数
        uncertainty_frame = ttk.LabelFrame(main_frame, text="不确定性分析参数", padding="10")
        uncertainty_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(uncertainty_frame, text="噪声实现次数:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(uncertainty_frame, from_=50, to=500, increment=50, 
                    textvariable=self.noise_realizations, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(uncertainty_frame, text="震源机制误差标准差 (度):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(uncertainty_frame, from_=1, to=15, increment=1, 
                    textvariable=self.mean_deviation, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # 反演参数
        inversion_frame = ttk.LabelFrame(main_frame, text="反演参数", padding="10")
        inversion_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(inversion_frame, text="迭代次数:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(inversion_frame, from_=3, to=20, increment=1, 
                    textvariable=self.n_iterations, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(inversion_frame, text="初始反演次数:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(inversion_frame, from_=5, to=50, increment=5, 
                    textvariable=self.n_realizations, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # 摩擦系数参数
        friction_frame = ttk.LabelFrame(main_frame, text="摩擦系数搜索范围", padding="10")
        friction_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(friction_frame, text="最小值:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(friction_frame, from_=0.2, to=0.8, increment=0.05, 
                    textvariable=self.friction_min, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(friction_frame, text="最大值:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        ttk.Spinbox(friction_frame, from_=0.6, to=1.5, increment=0.05, 
                    textvariable=self.friction_max, width=10).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(friction_frame, text="步长:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(friction_frame, from_=0.01, to=0.1, increment=0.01, 
                    textvariable=self.friction_step, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="开始运行", command=self.run_analysis, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="加载配置", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
    def browse_input_file(self):
        """浏览输入文件"""
        filename = filedialog.askopenfilename(
            title="选择震源机制数据文件",
            filetypes=[("数据文件", "*.dat *.txt"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir.set(dirname)
    
    def browse_figure_dir(self):
        """浏览图形目录"""
        dirname = filedialog.askdirectory(title="选择图形输出目录")
        if dirname:
            self.figure_dir.set(dirname)
    
    def validate_inputs(self):
        """验证输入参数"""
        if not self.input_file.get():
            messagebox.showerror("错误", "请选择输入文件")
            return False
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("错误", "输入文件不存在")
            return False
        
        if self.friction_min.get() >= self.friction_max.get():
            messagebox.showerror("错误", "摩擦系数最小值必须小于最大值")
            return False
        
        if self.deviation_good.get() >= self.deviation_bad.get():
            messagebox.showerror("错误", "偏差角'好'阈值必须小于'坏'阈值")
            return False
        
        return True
    
    def save_config(self):
        """保存配置到文件"""
        filename = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".config",
            filetypes=[("配置文件", "*.config"), ("所有文件", "*.*")]
        )
        if filename:
            import json
            config = {
                'input_file': self.input_file.get(),
                'output_dir': self.output_dir.get(),
                'figure_dir': self.figure_dir.get(),
                'instability_threshold': self.instability_threshold.get(),
                'deviation_good': self.deviation_good.get(),
                'deviation_bad': self.deviation_bad.get(),
                'noise_realizations': self.noise_realizations.get(),
                'mean_deviation': self.mean_deviation.get(),
                'n_iterations': self.n_iterations.get(),
                'n_realizations': self.n_realizations.get(),
                'friction_min': self.friction_min.get(),
                'friction_max': self.friction_max.get(),
                'friction_step': self.friction_step.get()
            }
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("成功", "配置已保存")
    
    def load_config(self):
        """从文件加载配置"""
        filename = filedialog.askopenfilename(
            title="加载配置文件",
            filetypes=[("配置文件", "*.config"), ("所有文件", "*.*")]
        )
        if filename:
            import json
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.input_file.set(config.get('input_file', ''))
                self.output_dir.set(config.get('output_dir', '../Output_improved'))
                self.figure_dir.set(config.get('figure_dir', '../Figures_improved'))
                self.instability_threshold.set(config.get('instability_threshold', 1.4))
                self.deviation_good.set(config.get('deviation_good', 20.0))
                self.deviation_bad.set(config.get('deviation_bad', 30.0))
                self.noise_realizations.set(config.get('noise_realizations', 100))
                self.mean_deviation.set(config.get('mean_deviation', 5.0))
                self.n_iterations.set(config.get('n_iterations', 6))
                self.n_realizations.set(config.get('n_realizations', 10))
                self.friction_min.set(config.get('friction_min', 0.40))
                self.friction_max.set(config.get('friction_max', 1.00))
                self.friction_step.set(config.get('friction_step', 0.05))
                
                messagebox.showinfo("成功", "配置已加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {str(e)}")
    
    def run_analysis(self):
        """运行分析"""
        if not self.validate_inputs():
            return
        
        # 保存参数并关闭窗口
        self.save_parameters()
        self.root.quit()
        self.root.destroy()
    
    def save_parameters(self):
        """保存参数供主程序使用"""
        global parameters
        parameters = {
            'input_file': self.input_file.get(),
            'output_dir': self.output_dir.get(),
            'figure_dir': self.figure_dir.get(),
            'instability_ratio_threshold': self.instability_threshold.get(),
            'deviation_angle_good': self.deviation_good.get(),
            'deviation_angle_bad': self.deviation_bad.get(),
            'N_noise_realizations': self.noise_realizations.get(),
            'mean_deviation': self.mean_deviation.get(),
            'N_iterations': self.n_iterations.get(),
            'N_realizations': self.n_realizations.get(),
            'friction_min': self.friction_min.get(),
            'friction_max': self.friction_max.get(),
            'friction_step': self.friction_step.get()
        }
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()
        return hasattr(self, 'parameters')

# 创建GUI并获取参数
def get_parameters_gui():
    """通过GUI获取参数"""
    gui = StressInversionGUI()
    gui.run()
    
    if 'parameters' in globals():
        return parameters
    else:
        print("用户取消了操作")
        sys.exit(0)

# 如果直接运行此文件，显示GUI
if __name__ == "__main__":
    params = get_parameters_gui()
    print("获取的参数：")
    for key, value in params.items():
        print(f"  {key}: {value}")
else:
    # 作为模块导入时，获取参数
    parameters = get_parameters_gui()
    
    # 设置参数
    input_file = parameters['input_file']
    output_dir = parameters['output_dir']
    figure_dir = parameters['figure_dir']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    # 基于输入文件名生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 输出文件
    output_file = os.path.join(output_dir, f'{base_name}_output')
    principal_mechanisms_file = os.path.join(output_dir, f'{base_name}_principal_mechanisms')
    
    # 图形文件
    shape_ratio_plot = os.path.join(figure_dir, f'{base_name}_shape_ratio')
    stress_plot = os.path.join(figure_dir, f'{base_name}_stress_directions')
    P_T_plot = os.path.join(figure_dir, f'{base_name}_P_T_axes')
    Mohr_plot = os.path.join(figure_dir, f'{base_name}_Mohr_circles')
    faults_plot = os.path.join(figure_dir, f'{base_name}_faults')
    
    # 其他参数
    instability_ratio_threshold = parameters['instability_ratio_threshold']
    deviation_angle_good = parameters['deviation_angle_good']
    deviation_angle_bad = parameters['deviation_angle_bad']
    N_noise_realizations = parameters['N_noise_realizations']
    mean_deviation = parameters['mean_deviation']
    N_iterations = parameters['N_iterations']
    N_realizations = parameters['N_realizations']
    friction_min = parameters['friction_min']
    friction_max = parameters['friction_max']
    friction_step = parameters['friction_step']
    
    # 形状比直方图参数
    shape_ratio_min = 0
    shape_ratio_max = 1
    shape_ratio_step = 0.025
    shape_ratio_axis = np.arange(shape_ratio_min+0.0125, shape_ratio_max, shape_ratio_step)