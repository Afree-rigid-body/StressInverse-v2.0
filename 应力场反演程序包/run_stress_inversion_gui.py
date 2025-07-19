#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图形界面版应力反演程序启动脚本
运行此脚本会打开参数设置窗口，然后自动运行应力反演分析
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("="*60)
    print("应力反演程序 - 图形界面版")
    print("="*60)
    print("\n正在启动参数设置窗口...\n")
    
    # 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行改进版主程序（会自动调用GUI）
    main_program = os.path.join(script_dir, "StressInverse_improved.py")
    
    try:
        # 使用当前Python解释器运行主程序
        result = subprocess.run([sys.executable, main_program], 
                              check=True, 
                              cwd=script_dir)
        
        if result.returncode == 0:
            print("\n程序运行完成！")
            print("请查看输出目录中的结果文件。")
        else:
            print("\n程序运行出错。")
            
    except subprocess.CalledProcessError as e:
        print(f"\n错误：程序运行失败")
        print(f"错误代码：{e.returncode}")
    except FileNotFoundError:
        print(f"\n错误：找不到主程序文件 {main_program}")
        print("请确保所有程序文件都在正确的位置。")
    except Exception as e:
        print(f"\n未知错误：{str(e)}")
    
    # 等待用户按键退出
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()