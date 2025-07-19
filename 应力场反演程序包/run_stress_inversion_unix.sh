#!/bin/bash
# 运行应力反演程序.sh
# Linux/Mac启动脚本 - 应力反演程序

echo "============================================================"
echo "应力反演程序 - 图形界面版"
echo "============================================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "错误：未找到Python！"
    echo "请先安装Python 3.6或更高版本"
    echo "Ubuntu/Debian: sudo apt-get install python3"
    echo "macOS: brew install python3"
    exit 1
fi

# 确定Python命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "使用Python: $($PYTHON_CMD --version)"
echo

# 检查必要的包
echo "检查依赖包..."
$PYTHON_CMD -c "import numpy, matplotlib, scipy, tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告：某些依赖包可能未安装"
    echo "建议运行: pip install numpy matplotlib scipy"
    echo
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 运行程序
echo "正在启动程序..."
echo

# 首先尝试运行GUI启动脚本
if [ -f "run_stress_inversion_gui.py" ]; then
    $PYTHON_CMD run_stress_inversion_gui.py
else
    # 如果启动脚本不存在，直接运行主程序
    echo "运行主程序..."
    $PYTHON_CMD StressInverse_improved.py
fi

echo
echo "程序运行结束"

# 如果是在终端中直接运行的，等待用户按键
if [ -t 0 ]; then
    read -p "按回车键退出..."
fi