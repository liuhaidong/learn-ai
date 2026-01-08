#!/bin/bash

echo "=== 自动化工程标书生成系统 - 快速安装 ==="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python3 --version || python --version || { echo "错误: 未找到 Python"; exit 1; }

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？ (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    
    echo "激活虚拟环境..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# 安装依赖
echo ""
echo "安装依赖包..."
pip install -r requirements.txt

# 配置环境变量
echo ""
echo "配置环境变量..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "已创建 .env 文件"
    echo "请编辑 .env 文件，填入你的 OPENAI_API_KEY"
else
    echo ".env 文件已存在"
fi

# 初始化示例数据
echo ""
read -p "是否初始化示例数据？ (y/n): " init_data
if [ "$init_data" = "y" ]; then
    python init_data.py
fi

echo ""
echo "=== 安装完成！ ==="
echo ""
echo "使用方法："
echo "1. 编辑 .env 文件，配置 OPENAI_API_KEY"
echo "2. 运行: python bid_agent.py --input 招标文件.docx --history 历史标书.docx"
echo ""
echo "测试模块:"
echo "python tests/test_modules.py"
