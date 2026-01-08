from typing import Dict


class CommercialAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_commercial_content(self, requirements: str) -> str:
        prompt = f"""
你现在是智能化弱电工程商务标书专家。

根据以下招标要求，生成商务部分内容：
{requirements}

需要包含：
1. 公司资质
2. 类似项目业绩
3. 项目团队介绍
4. 售后服务承诺
5. 报价明细（如需要）
"""
        return self.llm.generate(prompt)
