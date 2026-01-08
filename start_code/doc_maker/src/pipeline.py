import os
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger

sys.path.append(str(Path(__file__).parent / "src"))

from src.agents import LLMClient, BidAgent, CommercialAgent, TechnicalAgent, CostAgent, ComplianceAgent
from src.parsers import parse_docx, parse_docx_tables, parse_excel
from src.compliance import ComplianceChecker
from src.duplicate import DuplicateChecker
from src.pricing import PriceDatabase
from src.utils import AuditLogger
from src.utils.exporter import export_word


class BidGenerator:
    def __init__(self):
        self.llm = LLMClient()
        self.bid_agent = BidAgent(self.llm)
        self.compliance_checker = ComplianceChecker()
        self.dup_checker = DuplicateChecker()
        self.price_db = PriceDatabase()
        self.logger = AuditLogger()
        
        self.commercial_agent = CommercialAgent(self.llm)
        self.technical_agent = TechnicalAgent(self.llm)
        self.cost_agent = CostAgent(self.llm, self.price_db)
        self.compliance_agent = ComplianceAgent(self.llm, self.compliance_checker)
    
    def generate(self, bid_doc: str, history_docs: List[str] = None) -> Dict:
        logger.info(f"开始生成标书: {bid_doc}")
        
        # 解析招标文件
        req_text = "\n".join(parse_docx(bid_doc))
        req_tables = parse_docx_tables(bid_doc)
        
        # 解析历史标书
        history_texts = []
        if history_docs:
            for hdoc in history_docs:
                history_texts.extend(parse_docx(hdoc))
        
        self.logger.log("pipeline_start", {
            "bid_doc": bid_doc,
            "history_count": len(history_docs) if history_docs else 0
        })
        
        # 1. 生成规划
        logger.info("生成标书章节规划...")
        plan = self.bid_agent.generate_plan(req_text)
        logger.info(f"规划完成: {len(plan)} 字符")
        
        # 2. 定义章节
        chapters = [
            {"name": "项目概况", "type": "technical"},
            {"name": "技术方案", "type": "technical"},
            {"name": "视频监控系统", "type": "technical"},
            {"name": "入侵报警系统", "type": "technical"},
            {"name": "门禁系统", "type": "technical"},
            {"name": "综合布线", "type": "technical"},
            {"name": "网络系统", "type": "technical"},
            {"name": "施工组织设计", "type": "technical"},
            {"name": "设备材料表", "type": "technical"},
            {"name": "进度计划", "type": "technical"},
            {"name": "质量保证", "type": "technical"},
            {"name": "安全文明施工", "type": "technical"},
            {"name": "公司资质", "type": "commercial"},
            {"name": "类似项目业绩", "type": "commercial"},
            {"name": "项目团队", "type": "commercial"},
            {"name": "售后服务承诺", "type": "commercial"},
            {"name": "报价明细", "type": "cost"}
        ]
        
        chapter_outputs = {}
        
        for ch in chapters:
            logger.info(f"生成章节: {ch['name']}")
            
            # 生成草稿
            if ch["type"] == "technical":
                content = self.technical_agent.generate_technical_content(ch["name"], req_text)
            elif ch["type"] == "commercial":
                content = self.commercial_agent.generate_commercial_content(req_text)
            elif ch["type"] == "cost":
                content = self.cost_agent.generate_cost_estimate(req_text)
            else:
                content = self.bid_agent.generate_draft(ch["name"], plan, req_text)
            
            self.logger.log_generation(ch["name"], content, "draft")
            
            # 合规检查和修正
            compliance_result = self.compliance_agent.check_and_fix(ch["name"], content)
            if "fixed_text" in compliance_result:
                content = compliance_result["fixed_text"]
                logger.info(f"章节 {ch['name']} 已根据合规要求修正")
            
            self.logger.log_compliance_check(ch["name"], compliance_result)
            
            # 查重检查
            paragraphs = content.split("\n")
            dup_results = self.dup_checker.check_duplicates(paragraphs, history_texts)
            self.logger.log_duplicate_check(ch["name"], dup_results)
            
            chapter_outputs[ch["name"]] = content
        
        # 3. 合并并最终处理
        logger.info("合并章节并最终处理...")
        full_text = "\n\n".join([f"# {k}\n\n{v}" for k, v in chapter_outputs.items()])
        
        final_text = self.bid_agent.finalize_document(full_text)
        logger.info("最终处理完成")
        
        # 4. 导出 Word 文档
        output_path = "output/final_bid.docx"
        export_word(final_text, output_path)
        logger.info(f"标书已导出: {output_path}")
        
        # 5. 生成报告
        self.dup_checker.generate_report([], "output/dup_report.json")
        self.compliance_checker.generate_report(chapter_outputs, "output/compliance_report.json")
        
        result = {
            "status": "success",
            "output_file": output_path,
            "chapters": list(chapter_outputs.keys()),
            "full_text": final_text
        }
        
        self.logger.log("pipeline_complete", result)
        
        return result
