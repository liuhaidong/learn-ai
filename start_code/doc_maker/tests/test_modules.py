import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_price_db():
    from src.pricing import PriceDatabase
    
    print("测试单价数据库...")
    db = PriceDatabase("test_price_db.sqlite")
    
    # 添加测试数据
    db.add_price("网络摄像机", "400万像素", "台", 850.00, "供应商A")
    db.add_price("交换机", "16口千兆", "台", 1200.00, "供应商B")
    db.add_price("网线", "六类非屏蔽", "箱", 450.00, "市场价")
    
    # 查询测试
    result = db.query_price("网络摄像机", "400万像素")
    print(f"查询结果: {result}")
    
    print("单价数据库测试完成\n")


def test_compliance():
    from src.compliance import ComplianceChecker
    
    print("测试合规检查...")
    checker = ComplianceChecker("../data/rules/compliance_rules.json")
    
    test_text = """
    本项目视频监控系统包含摄像机数量50台，清晰度指标1080P，存储天数30天。
    网络结构采用星型拓扑，配置链路冗余。
    """
    
    result = checker.check_compliance("视频监控系统", test_text)
    print(f"合规检查结果: {result}")
    
    print("合规检查测试完成\n")


def test_duplicate():
    from src.duplicate import DuplicateChecker
    
    print("测试查重...")
    checker = DuplicateChecker()
    
    texts = [
        "本项目视频监控系统采用高清摄像机，覆盖主要出入口和关键区域。",
        "网络系统采用三层架构，核心交换机、汇聚交换机和接入交换机。"
    ]
    
    history = [
        "本项目视频监控系统采用高清摄像机，覆盖主要出入口和关键区域。",
        "网络系统采用三层架构，核心、汇聚和接入。"
    ]
    
    results = checker.check_duplicates(texts, history, threshold=0.75)
    
    for r in results:
        print(f"文本: {r['text'][:30]}...")
        print(f"相似度: {r['similarity']:.2f}")
        print(f"是否重复: {r['is_duplicate']}\n")
    
    print("查重测试完成\n")


def test_parsers():
    print("测试文档解析...")
    
    # 测试 Excel 解析
    from src.parsers import parse_excel
    
    test_file = "../data/prices/sample_prices.xlsx"
    if Path(test_file).exists():
        data = parse_excel(test_file)
        print(f"Excel 解析成功，读取 {len(data)} 行数据\n")
    else:
        print(f"测试文件不存在: {test_file}\n")
    
    print("文档解析测试完成\n")


if __name__ == "__main__":
    test_price_db()
    test_compliance()
    test_duplicate()
    test_parsers()
    
    print("所有测试完成！")
