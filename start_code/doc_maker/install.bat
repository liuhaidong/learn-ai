@echo off
echo === 自动化工程标书生成系统 - 快速安装 (Windows) ===
echo.

REM 检查 Python
python --version || (echo 错误: 未找到 Python & exit /b 1)

REM 创建虚拟环境
set /p create_venv="是否创建虚拟环境？ (y/n): "
if /i "%create_venv%"=="y" (
    echo 创建虚拟环境...
    python -m venv venv
    
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

REM 安装依赖
echo.
echo 安装依赖包...
pip install -r requirements.txt

REM 配置环境变量
echo.
echo 配置环境变量...
if not exist .env (
    copy .env.example .env
    echo 已创建 .env 文件
    echo 请编辑 .env 文件，填入你的 OPENAI_API_KEY
) else (
    echo .env 文件已存在
)

REM 初始化示例数据
echo.
set /p init_data="是否初始化示例数据？ (y/n): "
if /i "%init_data%"=="y" (
    python init_data.py
)

echo.
echo === 安装完成！ ===
echo.
echo 使用方法：
echo 1. 编辑 .env 文件，配置 OPENAI_API_KEY
echo 2. 运行: python bid_agent.py --input 招标文件.docx --history 历史标书.docx
echo.
echo 测试模块:
echo python tests\test_modules.py

pause
