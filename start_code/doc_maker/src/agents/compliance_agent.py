from typing import Dict
from ..compliance import ComplianceChecker


class ComplianceAgent:
    def __init__(self, llm_client, compliance_checker: ComplianceChecker):
        self.llm = llm_client
        self.compliance_checker = compliance_checker
    
    def generate_compliance_content(self, chapter_name: str, text: str) -> str:
        check_result = self.compliance_checker.check_compliance(chapter_name, text)
        
        if check_result["missing_items"]:
            prompt = f"""
以下内容缺少必要的合规条款：

缺失项：
{check_result["missing_items"]}

原内容：
{text}

请补充缺失的合规条款，确保标书符合相关标准。
"""
            return self.llm.generate(prompt)
        
        return text
    
    def check_and_fix(self, chapter_name: str, text: str) -> Dict:
        result = self.compliance_checker.check_compliance(chapter_name, text)
        
        if not result["compliance"]:
            fixed_text = self.generate_compliance_content(chapter_name, text)
            return {
                "chapter": chapter_name,
                "compliance": True,
                "fixed_text": fixed_text
            }
        
        return {
            "chapter": chapter_name,
            "compliance": True,
            "text": text
        }
