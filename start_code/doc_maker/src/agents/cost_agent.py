from typing import Dict
from ..pricing import PriceDatabase


class CostAgent:
    def __init__(self, llm_client, price_db: PriceDatabase):
        self.llm = llm_client
        self.price_db = price_db
    
    def generate_cost_estimate(self, requirements: str) -> str:
        prompt = f"""
你现在是智能化弱电工程造价专家。

根据以下招标要求，生成造价估算：
{requirements}

要求：
1. 设备清单清晰
2. 单价合理
3. 报价结构完整
"""
        
        base_content = self.llm.generate(prompt)
        
        # TODO: 查询数据库补充价格
        return base_content
    
    def query_material_price(self, material: str, spec: str = None):
        return self.price_db.query_price(material, spec)
