from .llm_client import LLMClient
from typing import Dict


class BidAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_plan(self, requirements: str) -> str:
        prompt = f"""
你是智能化弱电工程标书架构师。

根据以下招标要求生成《标书章节规划》：
{requirements}

输出格式：
1. 章节大纲
2. 每章任务目标
3. 需要引用的标准
4. 风险点与需要补充的信息
"""
        return self.llm.generate(prompt)
    
    def generate_draft(self, chapter_title: str, plan: str, requirements: str) -> str:
        prompt = f"""
章节：{chapter_title}

根据规划：
{plan}

和招标要求：
{requirements}

生成该章节的草稿，确保数据不编造。
"""
        return self.llm.generate(prompt)
    
    def refine_with_standards(self, text: str, standards: str, history: str) -> str:
        prompt = f"""
以下为草稿内容：
{text}

请参考行业标准条款：
{standards}

以及历史标书内容：
{history}

增强草稿内容，保持专业性。
"""
        return self.llm.generate(prompt)
    
    def finalize_document(self, full_text: str) -> str:
        prompt = f"""
请将全文进行最终处理：
- 术语一致
- 编号格式统一
- 工程化书面表达

文本：
{full_text}
"""
        return self.llm.generate(prompt)
    
    def comparative_rewrite(self, requirement: str, generated_text: str) -> str:
        prompt = f"""
对比 A（招标要求）和 B（生成文本）。

A：
{requirement}

B：
{generated_text}

列出不一致点，并执行安全改写，使 B 完全对齐招标内容。
不允许出现招标文件中不存在的数据或承诺。
"""
        return self.llm.generate(prompt)
