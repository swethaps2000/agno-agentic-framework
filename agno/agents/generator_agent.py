# agents/generator_agent.py
import os
import re
from typing import Dict, List
import google.generativeai as genai
from bson import ObjectId
from datetime import datetime
from db.db_setup import setup_mongo
from agents.base_agent import agent
from dotenv import load_dotenv

load_dotenv()

@agent(name="GeneratorAgent")
class GeneratorAgent:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
        self.mongo_colls = setup_mongo()

    def run(self, user_id: str, query: str, fused_context: str) -> Dict:
        answer_text = self._generate_gemini_answer(query, fused_context)
        
        citations = self._extract_citations(answer_text)
        
        all_candidate_ids = re.findall(r'TX_([a-f0-9]{24})', fused_context)  # Extract from context
        final_citations = list(set(citations + all_candidate_ids))
        
        verified_total = self._verify_totals_from_db(user_id, final_citations)
        
        if all_candidate_ids:
            summary = f"\n\n📊 Summary: Found {len(all_candidate_ids)} relevant transactions totaling ₹{verified_total:,.2f}"
            answer_text += summary
        
        return {
            'answer': answer_text,
            'citations': final_citations,
            'verified_total': verified_total
        }
    
    def _generate_gemini_answer(self, query: str, context: str) -> str:
        if not self.gemini_model:
            return "I found relevant transactions based on your query. Please check the verified totals and citations below."
        
        prompt = f"""You are a personal finance assistant analyzing transaction data. Answer the user's query with specific details and insights.

IMPORTANT:
1. Reference specific transactions using format [TX_<id>] when mentioning amounts
2. Provide concrete numbers and calculations
3. Give actionable financial insights
4. Be concise but comprehensive
5. Show calculations when computing totals
6. Identify spending patterns when relevant

USER QUERY: {query}

TRANSACTIONS: {context}

Provide detailed, helpful response using the transaction data."""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini generation failed: {e}")
            return f"Found relevant transactions for your query. Verified totals below. (API Error: {str(e)})"
    
    def _extract_citations(self, answer_text: str) -> List[str]:
        pattern = r'(?:\[)?TX_([a-f0-9]{24})(?:\])?'
        return re.findall(pattern, answer_text, re.IGNORECASE)
    
    def _verify_totals_from_db(self, user_id: str, cited_ids: List[str]) -> float:
        total = 0.0
        
        for tx_id in cited_ids:
            try:
                doc = self.mongo_colls['expense_coll'].find_one({'_id': ObjectId(tx_id)})
                if not doc:
                    doc = self.mongo_colls['income_coll'].find_one({'_id': ObjectId(tx_id)})
                
                if doc and doc.get('user_id') == user_id:
                    amount = float(doc.get('amount', 0))
                    total += amount
            except Exception as e:
                print(f"Error verifying transaction {tx_id}: {e}")
                continue
        
        return total