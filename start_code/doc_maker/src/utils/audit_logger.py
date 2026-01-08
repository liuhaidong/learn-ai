import json
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class AuditLogger:
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, event_type: str, data: Dict[str, Any]) -> str:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data
        }
        
        log_file = os.path.join(self.log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        return log_file
    
    def log_generation(self, chapter: str, content: str, status: str):
        self.log("generation", {
            "chapter": chapter,
            "status": status,
            "length": len(content)
        })
    
    def log_compliance_check(self, chapter: str, result: Dict):
        self.log("compliance_check", {
            "chapter": chapter,
            "result": result
        })
    
    def log_duplicate_check(self, chapter: str, result: list):
        self.log("duplicate_check", {
            "chapter": chapter,
            "result": result
        })
