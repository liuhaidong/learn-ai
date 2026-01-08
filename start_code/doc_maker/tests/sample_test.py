#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))


def sample_generation():
    """测试完整的标书生成流程（使用示例文本）"""
    print("=== 测试标书生成流程 ===\n")
    
    from src.pipeline import BidGenerator
    
    # 创建生成器（不会调用真实 API）
    print("初始化生成器...")
    print("提示: 如果未配置 OPENAI_API_KEY，将返回错误信息\n")
    
    generator = BidGenerator()
    
    # 示例招标要求
    sample_requirements = """
    项目名称：XX办公楼智能化弱电工程
    
    建设内容：
    1. 视频监控系统：共需安装摄像机50台，覆盖主要出入口和公共区域
    2. 门禁系统：10个门禁点位，支持指纹和卡片开锁
    3. 综合布线：信息点数量500个，采用六类布线系统
    4. 网络系统：三层网络架构，核心交换机1台，接入交换机10台
    
    技术要求：
    - 符合GB 50348-2018《安全防范工程技术标准》
    - 符合GB 50311-2016《综合布线系统工程设计规范》
    - 视频存储不少于30天
    - 网络带宽不低于1000Mbps
    """
    
    print("准备生成标书...")
    print("提示: 实际运行请使用 python bid_agent.py --input <招标文件.docx>\n")
    
    # 保存示例要求到文件
    output_dir = Path("data/bids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = output_dir / "sample_requirement.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_requirements)
    
    print(f"示例要求已保存到: {sample_file}")
    print("\n请准备实际的 .docx 招标文件，然后运行:")
    print("python bid_agent.py --input 招标文件.docx --history 历史标书1.docx --history 历史标书2.docx")


if __name__ == "__main__":
    sample_generation()
