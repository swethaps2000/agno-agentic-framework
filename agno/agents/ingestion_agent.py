# agents/ingestion_agent.py
import os
import json
import re
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from db.db_setup import setup_mongo, setup_chroma
from agents.base_agent import agent
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Transaction:
    id: Optional[str] = None
    user_id: str = ""
    amount: float = 0.0
    type: str = "expense"
    date: str = ""
    category: str = "uncategorized"
    description: str = ""
    notes: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@agent(name="IngestionAgent")
class IngestionAgent:
    def __init__(self):
        self.mongo_colls = setup_mongo()
        self.chroma_collection = setup_chroma()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None

    async def run(self, user_id: str, text: str) -> Dict:
        raw_record = {
            "user_id": user_id,
            "text": text,
            "created_at": datetime.utcnow()
        }
        raw_id = self.mongo_colls['transactions_raw'].insert_one(raw_record).inserted_id
        
        parsed = self._parse_transaction(text)
        
        transaction = Transaction(
            user_id=user_id,
            amount=float(parsed.get('amount', 0.0)),
            type=parsed.get('type', 'expense'),
            date=parsed.get('date', datetime.now().strftime('%Y-%m-%d')),
            category=parsed.get('category', 'uncategorized'),
            description=parsed.get('description', text),
            notes=text
        )
        
        tx_id = await self._store_transaction(transaction)
        
        await self._vectorize_transaction(tx_id, transaction)
        
        return {
            'raw_id': str(raw_id),
            'tx_id': tx_id,
            'parsed': parsed
        }
    
    def _parse_transaction(self, text: str) -> Dict:
        if self.gemini_model:
            return self._gemini_parse(text)
        else:
            return self._heuristic_parse(text)
    
    def _gemini_parse(self, text: str) -> Dict:
        prompt = f"""Parse this financial transaction text into structured JSON format. Extract:

- amount: The monetary amount (as number, no currency symbols)
- currency: Currency code (INR for Indian Rupees, USD, etc.)
- date: Date in YYYY-MM-DD format (use today if not specified: {datetime.now().strftime('%Y-%m-%d')})
- category: Spending category (groceries, rent, salary, fuel, shopping, dining, bills, medical, investment, etc.)
- type: Either "income" or "expense"
- description: Clean, concise description

Transaction text: "{text}"

Return ONLY valid JSON object with these fields. No explanations.

Example: {{"amount": 5000, "currency": "INR", "date": "2025-01-15", "category": "groceries", "type": "expense", "description": "Grocery shopping at supermarket"}}"""

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text.rsplit('\n', 1)[0]
            
            parsed = json.loads(response_text)
            
            required_fields = ['amount', 'currency', 'date', 'category', 'type', 'description']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            if isinstance(parsed['amount'], str):
                parsed['amount'] = float(re.sub(r'[^\d.]', '', parsed['amount']))
                
            return parsed
            
        except Exception as e:
            print(f"Gemini parsing failed: {e}. Using heuristics.")
            return self._heuristic_parse(text)
    
    def _heuristic_parse(self, text: str) -> Dict:
        amount = 0.0
        amount_match = re.search(r'[₹Rs\s]*(\d+[\,\d]*)', text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))
        
        income_keywords = ['received', 'salary', 'got', 'credited', 'income', 'payment', 'bonus']
        tx_type = 'income' if any(w in text.lower() for w in income_keywords) else 'expense'
        
        category_map = {
            'rent': ['rent', 'house rent'],
            'groceries': ['grocery', 'groceries', 'vegetables', 'supermarket'],
            'fuel': ['fuel', 'petrol', 'gas'],
            'food': ['restaurant', 'dining', 'food', 'pizza', 'coffee'],
            'bills': ['electricity', 'water', 'internet', 'mobile', 'recharge'],
            'shopping': ['shopping', 'amazon', 'clothes', 'electronics'],
            'transport': ['uber', 'taxi', 'metro', 'travel'],
            'medical': ['medical', 'doctor', 'hospital', 'medicine'],
            'investment': ['investment', 'mutual fund', 'sip'],
            'salary': ['salary'],
            'entertainment': ['movie', 'tickets', 'gym']
        }
        
        category = 'uncategorized'
        for cat, keywords in category_map.items():
            if any(keyword in text.lower() for keyword in keywords):
                category = cat
                break
        
        return {
            'amount': amount,
            'currency': 'INR',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': category,
            'type': tx_type,
            'description': text
        }
    
    def _get_default_value(self, field: str):
        defaults = {
            'amount': 0.0,
            'currency': 'INR',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': 'uncategorized',
            'type': 'expense',
            'description': 'Transaction'
        }
        return defaults.get(field, '')
    
    async def _store_transaction(self, transaction: Transaction) -> str:
        record = {
            'user_id': transaction.user_id,
            'amount': transaction.amount,
            'type': transaction.type,
            'date': transaction.date,
            'category': transaction.category,
            'notes': transaction.notes,
            'description': transaction.description,
            'created_at': datetime.utcnow()
        }
        
        if transaction.type == 'income':
            result = self.mongo_colls['income_coll'].insert_one(record)
        else:
            result = self.mongo_colls['expense_coll'].insert_one(record)
            
        return str(result.inserted_id)
    
    async def _vectorize_transaction(self, tx_id: str, transaction: Transaction):
        enhanced_text = f"{transaction.description} | Amount: ₹{transaction.amount} | Type: {transaction.type} | Category: {transaction.category} | Date: {transaction.date}"
        
        metadata = {
            'user_id': transaction.user_id,
            'type': transaction.type,
            'category': transaction.category,
            'amount': float(transaction.amount),
            'date': transaction.date
        }
        
        embedding = self.embedding_model.encode([enhanced_text])[0]
        
        self.chroma_collection.upsert(
            ids=[tx_id],
            metadatas=[metadata],
            documents=[enhanced_text],
            embeddings=[embedding.tolist()]
        )