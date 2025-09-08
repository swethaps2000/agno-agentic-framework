# agents/fusion_agent.py
from typing import List, Dict
from agents.base_agent import agent

@agent(name="FusionAgent")
class FusionAgent:
    def run(self, candidates: List[Dict]) -> str:
        return self._build_context_passages(candidates)
    
    def _build_context_passages(self, candidates: List[Dict]) -> str:
        parts = []
        for c in candidates:
            metadata = c.get('metadata', {})
            tx_id = c.get('id')
            text = c.get('text', '')
            
            amount = metadata.get('amount', 'N/A')
            tx_type = metadata.get('type', 'unknown')
            category = metadata.get('category', 'uncategorized')
            date = metadata.get('date', 'unknown date')
            
            parts.append(f"[TX_{tx_id}] {text} | Amount: ₹{amount} | Type: {tx_type} | Category: {category} | Date: {date}")
        
        return "\n".join(parts)