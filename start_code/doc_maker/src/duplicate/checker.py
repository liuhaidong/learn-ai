import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateChecker:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def _similarity(self, text_a: str, text_b: str) -> float:
        v1 = self.model.encode(text_a, convert_to_numpy=True)
        v2 = self.model.encode(text_b, convert_to_numpy=True)
        sim = cosine_similarity([v1], [v2])[0][0]
        return float(sim)
    
    def check_duplicates(
        self, 
        texts: List[str], 
        history_texts: List[str], 
        threshold: float = 0.75
    ) -> List[Dict]:
        results = []
        for t in texts:
            best_sim, best_src = 0.0, None
            
            for h in history_texts:
                sim = self._similarity(t, h)
                if sim > best_sim:
                    best_sim, best_src = sim, h
            
            results.append({
                "text": t,
                "similarity": best_sim,
                "source": best_src if best_sim > threshold else None,
                "is_duplicate": best_sim > threshold
            })
        return results
    
    def generate_report(self, results: List[Dict], output_path: str = "data/logs/dup_report.json"):
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
