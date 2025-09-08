# agents/rerank_agent.py
from typing import List, Dict
from sentence_transformers import CrossEncoder
from agents.base_agent import agent

@agent(name="RerankAgent")
class RerankAgent:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def run(self, query: str, candidates: List[Dict]) -> List[Dict]:
        return self.rerank_candidates(query, candidates, top_k=5)
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []
        
        try:
            pairs = [(query, c['text']) for c in candidates]
            
            scores = self.reranker.predict(pairs)
            
            for candidate, score in zip(candidates, scores):
                candidate['score'] = float(score)
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:top_k]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return candidates[:top_k]