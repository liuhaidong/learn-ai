from typing import Dict


class TechnicalAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_technical_content(self, chapter_name: str, requirements: str) -> str:
        prompt = f"""
你现在是智能化弱电工程技术标书专家。

章节：{chapter_name}

根据以下招标要求，生成该章节技术内容：
{requirements}

要求：
1. 技术方案清晰完整
2. 符合行业标准
3. 图文并茂（用文字描述图表）
4. 具有可实施性
"""
        return self.llm.generate(prompt)
