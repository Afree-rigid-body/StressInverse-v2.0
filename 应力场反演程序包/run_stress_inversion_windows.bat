@echo off
REM 运行应力反演程序.bat
REM Windows批处理脚本 - 启动应力反演程序

echo ============================================================
echo 应力反演程序 - 图形界面版
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python！
    echo 请先安装Python 3.6或更高版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 正在启动程序...
echo.

REM 运行主程序
python run_stress_inversion_gui.py

REM 如果上面的脚本不存在，尝试直接运行改进版主程序
if errorlevel 1 (
    echo 尝试直接运行主程序...
    python StressInverse_improved.py
)

echo.
echo 程序运行结束
pause