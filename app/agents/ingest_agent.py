from datetime import datetime
from app.db.mongodb import transactions_raw, income_coll, expense_coll
from app.vectorstore.chroma_store import upsert_transaction
# from vectorstore import chroma_store


from dotenv import load_dotenv
import os, json
load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# NOTE: replace the following with actual Gemini client calls. We implement a simple placeholder


# Simple heuristics fallback parser in case LLM parsing is unavailable
import re


def heuristic_parse(text: str):
    # find amount like ₹500 or 500
    amt = 0.0
    m = re.search(r'[₹Rs\s]*(\d+[\,\d]*)', text)
    if m:
        amt = float(m.group(1).replace(',', '').strip())
        # find date keywords
    date = datetime.utcnow().date()
    if 'today' in text.lower():
        date = datetime.utcnow().date()
        # naive type detection
    t = 'income' if any(w in text.lower() for w in ['received','salary','got','credited']) else 'expense'
        # category: words after 'on' or 'for'
    cat = 'uncategorized'
    m2 = re.search(r'on\s+(\w+)', text.lower())
    if m2:
        cat = m2.group(1)
    return {
        'amount': amt,
        'currency': 'INR',
        'date': str(date),
        'category': cat,
        'type': t,
        'notes': text
        }


async def ingest_transaction(user_id: str, text: str):
    raw = {"user_id":user_id, "text":text, "created_at": datetime.utcnow()}
    raw_id = transactions_raw.insert_one(raw).inserted_id


    # try LLM parse here (placeholder). If GEMINI_API_KEY present, call Gemini parse; else use heuristic
    parsed = heuristic_parse(text)


    # normalize
    amount = parsed.get('amount', 0.0)
    dtype = parsed.get('type', 'expense')
    date = parsed.get('date')
    category = parsed.get('category', 'uncategorized')


    rec = {
        'user_id': user_id,
        'amount': amount,
        'type': dtype,
        'date': date,
        'category': category,
        'notes': text,
        'created_at': datetime.utcnow()
    }

    # Insert into the appropriate collection
    if dtype == 'income':
        tx_id = income_coll.insert_one(rec).inserted_id
    else:
        tx_id = expense_coll.insert_one(rec).inserted_id
    metadata = {
        'user_id': user_id,
        'type': dtype,
        'category': category,
        'amount': amount,
        'date': date
    }
    # Optionally upsert into vectorstore
    await upsert_transaction(str(tx_id), rec,metadata)

    return {'raw_id': str(raw_id), 'tx_id': str(tx_id), 'parsed': parsed}