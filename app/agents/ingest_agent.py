# ===== FILE: app/agents/ingest_agent.py =====
from datetime import datetime
import json
import re
from app.db.mongodb import transactions_raw, income_coll, expense_coll
from app.vectorstore.chroma_store import upsert_transaction
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

def gemini_parse_transaction(text: str) -> dict:
    """Use Gemini Flash to parse transaction text"""
    if not model:
        return heuristic_parse(text)
    
    prompt = f"""Parse this financial transaction text into structured JSON format. Extract the following information:

- amount: The monetary amount (as a number, no currency symbols)
- currency: The currency code (INR for Indian Rupees, USD, etc.)
- date: Date in YYYY-MM-DD format (use today's date if not specified: {datetime.now().strftime('%Y-%m-%d')})
- category: Spending category (groceries, rent, salary, fuel, shopping, dining, bills, medical, investment, etc.)
- type: Either "income" or "expense"
- description: A clean, concise description of the transaction

Transaction text: "{text}"

Return ONLY a valid JSON object with these fields. Do not include any other text or explanations.

Example format:
{{"amount": 5000, "currency": "INR", "date": "2025-01-15", "category": "groceries", "type": "expense", "description": "Grocery shopping at supermarket"}}"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            response_text = response_text.rsplit('\n', 1)[0]
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Validate required fields
        required_fields = ['amount', 'currency', 'date', 'category', 'type', 'description']
        for field in required_fields:
            if field not in parsed:
                parsed[field] = get_default_value(field)
        
        # Ensure amount is numeric
        if isinstance(parsed['amount'], str):
            parsed['amount'] = float(re.sub(r'[^\d.]', '', parsed['amount']))
        
        return parsed
        
    except Exception as e:
        print(f"Gemini parsing failed: {e}. Falling back to heuristic parsing.")
        return heuristic_parse(text)

def get_default_value(field: str):
    """Get default values for missing fields"""
    defaults = {
        'amount': 0.0,
        'currency': 'INR',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'category': 'uncategorized',
        'type': 'expense',
        'description': 'Transaction'
    }
    return defaults.get(field, '')

def heuristic_parse(text: str) -> dict:
    """Fallback heuristic parser for when Gemini is unavailable"""
    # Find amount like ₹500 or 500
    amt = 0.0
    m = re.search(r'[₹Rs\s]*(\d+[\,\d]*)', text)
    if m:
        amt = float(m.group(1).replace(',', '').strip())
    
    # Find date keywords
    date = datetime.now().strftime('%Y-%m-%d')
    if 'today' in text.lower():
        date = datetime.now().strftime('%Y-%m-%d')
    elif 'yesterday' in text.lower():
        date = (datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Date pattern matching (basic)
    date_patterns = [
        r'(\d{1,2})[st|nd|rd|th]*\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                if 'January|February' in pattern:  # Month name format
                    day, month, year = match.groups()
                    month_num = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04',
                        'may': '05', 'june': '06', 'july': '07', 'august': '08',
                        'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }.get(month.lower(), '01')
                    date = f"{year}-{month_num.zfill(2)}-{day.zfill(2)}"
                elif '/' in pattern:  # MM/DD/YYYY format
                    month, day, year = match.groups()
                    date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:  # YYYY-MM-DD format
                    date = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            except:
                pass
    
    # Naive type detection
    income_keywords = ['received', 'salary', 'got', 'credited', 'income', 'payment', 'bonus', 'dividend', 'freelance', 'consulting']
    tx_type = 'income' if any(w in text.lower() for w in income_keywords) else 'expense'
    
    # Category detection
    category_map = {
        'rent': ['rent', 'house rent'],
        'groceries': ['grocery', 'groceries', 'vegetables', 'supermarket'],
        'fuel': ['fuel', 'petrol', 'gas'],
        'food': ['restaurant', 'dining', 'food', 'pizza', 'coffee'],
        'bills': ['electricity', 'water', 'internet', 'broadband', 'mobile', 'recharge'],
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
        'amount': amt,
        'currency': 'INR',
        'date': date,
        'category': category,
        'type': tx_type,
        'description': text
    }

async def ingest_transaction(user_id: str, text: str):
    """Ingest and parse transaction using Gemini Flash"""
    # Store raw transaction
    raw = {"user_id": user_id, "text": text, "created_at": datetime.utcnow()}
    raw_id = transactions_raw.insert_one(raw).inserted_id
    
    # Parse transaction using Gemini or fallback
    parsed = gemini_parse_transaction(text)
    
    # Normalize data
    amount = float(parsed.get('amount', 0.0))
    dtype = parsed.get('type', 'expense')
    date = parsed.get('date', datetime.now().strftime('%Y-%m-%d'))
    category = parsed.get('category', 'uncategorized')
    description = parsed.get('description', text)
    
    # Create database record
    rec = {
        'user_id': user_id,
        'amount': amount,
        'type': dtype,
        'date': date,
        'category': category,
        'notes': text,  # Keep original text
        'description': description,  # Cleaned description from Gemini
        'created_at': datetime.utcnow()
    }
    
    # Insert into appropriate collection
    if dtype == 'income':
        tx_id = income_coll.insert_one(rec).inserted_id
    else:
        tx_id = expense_coll.insert_one(rec).inserted_id
    
    # Prepare metadata for vector store
    metadata = {
        'user_id': user_id,
        'type': dtype,
        'category': category,
        'amount': amount,
        'date': date
    }
    
    # Upsert into vector store with enhanced text
    enhanced_text = f"{description} | Amount: ₹{amount} | Type: {dtype} | Category: {category} | Date: {date}"
    await upsert_transaction(str(tx_id), {'notes': enhanced_text}, metadata)
    
    return {
        'raw_id': str(raw_id), 
        'tx_id': str(tx_id), 
        'parsed': parsed
    }