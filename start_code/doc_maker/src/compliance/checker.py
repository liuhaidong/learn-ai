import json
from typing import List, Dict
from pathlib import Path


class ComplianceChecker:
    def __init__(self, rules_path: str = "data/rules/compliance_rules.json"):
        self.rules_path = rules_path
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[Dict]:
        path = Path(self.rules_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("chapters", [])
        return []
    
    def check_compliance(self, chapter_name: str, text: str) -> Dict:
        missing = []
        warnings = []
        
        for chapter in self.rules:
            if chapter["name"] == chapter_name:
                for rule in chapter["rules"]:
                    for item in rule["must_include"]:
                        if item not in text:
                            missing.append({
                                "rule": rule["id"],
                                "missing": item,
                                "reference": rule["reference"]
                            })
                break
        
        return {
            "chapter": chapter_name,
            "compliance": len(missing) == 0,
            "missing_items": missing,
            "warnings": warnings
        }
    
    def check_full_document(self, document: Dict[str, str]) -> Dict:
        results = {}
        for chapter, content in document.items():
            results[chapter] = self.check_compliance(chapter, content)
        return results
    
    def generate_report(self, results: Dict, output_path: str = "data/logs/compliance_report.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
