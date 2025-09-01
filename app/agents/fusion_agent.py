# ===== FILE: app/agents/fusion_agent.py =====
from app.db.mongodb import income_coll, expense_coll
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Placeholder: implement actual Gemini calls with `google-genai` SDK.


# build a concise context prompt and call Gemini (placeholder). We'll compute numeric checks locally.


def build_context_passages(candidates: list):
    parts = []
    for c in candidates:
        md = c.get('metadata', {})
        txid = c.get('id')
        text = c.get('text')
        parts.append(f"[{txid}] {text}")
    return "\n".join(parts)




def verify_totals_from_db(user_id: str, cited_ids: list):
    # compute sums for cited transaction ids directly from MongoDB
    from bson import ObjectId
    total = 0.0
    for tid in cited_ids:
        try:
            doc = expense_coll.find_one({'_id': ObjectId(tid)}) or income_coll.find_one({'_id': ObjectId(tid)})
            if doc:
                total += float(doc.get('amount', 0))
        except Exception:
            continue
    return total




def generate_answer(user_id: str, query: str, candidates: list, want_plot: bool = False):
    # select top passages and build prompt
    ctx = build_context_passages(candidates)
    system = (
    "You are a personal finance assistant. Use the passages to answer the user's query. "
    "For any numeric fact, include inline citations like [TX_<id>]. If you compute totals, verify them."
    )


    prompt = f"{system}\nUSER QUERY: {query}\nCONTEXT:\n{ctx}\nAnswer concisely."


    # Placeholder: instead of calling Gemini, we synthesize a short answer using local heuristics.
    # In production, call Gemini with `prompt` and parse the response.
    answer_text = "(Generator placeholder) I found relevant transactions. See citations in the passages."


    # collect cited ids
    cited_ids = [c['id'] for c in candidates]
    verified_total = verify_totals_from_db(user_id, cited_ids)


    return {
    'answer_text': answer_text,
    'citations': cited_ids,
    'verified_total': verified_total,
    'plot_path': None
    }