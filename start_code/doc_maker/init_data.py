#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))


def init_sample_data():
    """初始化示例数据"""
    from src.pricing import PriceDatabase
    from src.utils import AuditLogger
    
    print("=== 初始化示例数据 ===\n")
    
    # 初始化单价数据库
    print("1. 初始化单价数据库...")
    db = PriceDatabase()
    
    materials = [
        ("网络摄像机", "400万像素", "台", 850.00, "供应商A"),
        ("网络摄像机", "200万像素", "台", 650.00, "供应商A"),
        ("网络摄像机", "4K超清", "台", 1500.00, "供应商B"),
        ("交换机", "16口千兆", "台", 1200.00, "供应商B"),
        ("交换机", "24口千兆", "台", 1800.00, "供应商B"),
        ("交换机", "核心交换机", "台", 15000.00, "供应商C"),
        ("网线", "六类非屏蔽", "箱", 450.00, "市场价"),
        ("网线", "超五类非屏蔽", "箱", 320.00, "市场价"),
        ("光纤", "4芯单模", "米", 2.50, "供应商D"),
        ("门禁控制器", "单门", "台", 800.00, "供应商E"),
        ("门禁控制器", "双门", "台", 1200.00, "供应商E"),
        ("读卡器", "IC卡", "个", 150.00, "供应商E"),
        ("UPS电源", "3KVA", "台", 2500.00, "供应商F"),
        ("精密空调", "5匹", "台", 15000.00, "供应商F"),
        ("机柜", "42U", "台", 1800.00, "供应商G"),
        ("监控硬盘", "4TB", "块", 800.00, "供应商H"),
        ("监控硬盘", "8TB", "块", 1200.00, "供应商H"),
        ("NVR录像机", "32路", "台", 3500.00, "供应商H"),
        ("报警主机", "8防区", "台", 600.00, "供应商I"),
        ("红外对射", "200米", "对", 350.00, "供应商I")
    ]
    
    for mat in materials:
        db.add_price(*mat)
    
    print(f"已添加 {len(materials)} 条单价记录\n")
    
    # 初始化审计日志目录
    print("2. 初始化审计日志...")
    logger = AuditLogger()
    logger.log("init", {"message": "系统初始化完成"})
    print("审计日志已初始化\n")
    
    # 创建必要的目录结构
    dirs = ["data/bids", "data/history", "data/rules", "data/prices", "data/logs", "output"]
    print("3. 创建目录结构...")
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"已创建 {len(dirs)} 个目录\n")
    
    print("=== 初始化完成 ===")
    print("\n系统已准备就绪！")
    print("运行测试: python tests/test_modules.py")
    print("运行生成: python bid_agent.py --input <招标文件.docx>")


if __name__ == "__main__":
    init_sample_data()
