#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安装应力反演程序所需的依赖包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ {package} 安装失败")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("="*60)
    print("应力反演程序 - 依赖包安装")
    print("="*60)
    print()
    
    # 需要安装的包
    packages = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib', 
        'scipy': 'scipy',
        'tkinter': None  # tkinter通常随Python一起安装
    }
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    if sys.version_info < (3, 6):
        print("警告：建议使用Python 3.6或更高版本")
    print()
    
    # 检查和安装包
    print("检查依赖包...")
    need_install = []
    
    for import_name, install_name in packages.items():
        if import_name == 'tkinter':
            # 特殊处理tkinter
            try:
                import tkinter
                print(f"✓ {import_name} 已安装")
            except ImportError:
                print(f"✗ {import_name} 未安装")
                print("  tkinter通常随Python一起安装。")
                print("  Ubuntu/Debian: sudo apt-get install python3-tk")
                print("  macOS: 通常已包含在Python中")
                print("  Windows: 通常已包含在Python中")
        else:
            if check_package(import_name):
                print(f"✓ {import_name} 已安装")
            else:
                print(f"✗ {import_name} 未安装")
                need_install.append(install_name)
    
    print()
    
    # 安装缺失的包
    if need_install:
        print(f"需要安装的包: {', '.join(need_install)}")
        answer = input("是否开始安装？(y/n): ").lower()
        
        if answer == 'y':
            print()
            # 首先升级pip
            print("升级pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
                print("✓ pip升级成功")
            except:
                print("✗ pip升级失败，继续安装其他包...")
            
            print()
            
            # 安装包
            success = []
            failed = []
            
            for package in need_install:
                if install_package(package):
                    success.append(package)
                else:
                    failed.append(package)
            
            print()
            print("安装完成！")
            if success:
                print(f"成功安装: {', '.join(success)}")
            if failed:
                print(f"安装失败: {', '.join(failed)}")
                print("请手动安装失败的包或查看错误信息")
        else:
            print("取消安装")
    else:
        print("所有必需的包都已安装！")
    
    print()
    
    # 测试导入
    print("测试导入...")
    all_ok = True
    
    for package in ['numpy', 'matplotlib', 'scipy']:
        try:
            __import__(package)
            print(f"✓ {package} 导入成功")
        except ImportError as e:
            print(f"✗ {package} 导入失败: {e}")
            all_ok = False
    
    # 测试matplotlib后端
    try:
        import matplotlib
        matplotlib.use('Agg')
        print("✓ matplotlib后端设置成功")
    except Exception as e:
        print(f"✗ matplotlib后端设置失败: {e}")
        all_ok = False
    
    print()
    if all_ok:
        print("恭喜！所有依赖包都已正确安装。")
        print("您现在可以运行应力反演程序了。")
    else:
        print("某些包存在问题，请检查错误信息。")
    
    print()
    input("按回车键退出...")

if __name__ == "__main__":
    main()