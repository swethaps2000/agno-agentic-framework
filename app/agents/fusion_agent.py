# ===== FILE: app/agents/fusion_agent.py =====
from app.db.mongodb import income_coll, expense_coll
from datetime import datetime
import os
import google.generativeai as genai
import re
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("Warning: GEMINI_API_KEY not found. Using fallback responses.")

def build_context_passages(candidates: List[Dict]) -> str:
    """Build context from retrieved candidates"""
    parts = []
    for c in candidates:
        md = c.get('metadata', {})
        txid = c.get('id')
        text = c.get('text', '')
        
        # Extract key info from metadata
        amount = md.get('amount', 'N/A')
        tx_type = md.get('type', 'unknown')
        category = md.get('category', 'uncategorized')
        date = md.get('date', 'unknown date')
        
        # Format transaction info
        parts.append(f"[TX_{txid}] {text} | Amount: ₹{amount} | Type: {tx_type} | Category: {category} | Date: {date}")
    
    return "\n".join(parts)

def verify_totals_from_db(user_id: str, cited_ids: List[str]) -> float:
    """Compute sums for cited transaction ids directly from MongoDB"""
    from bson import ObjectId
    total = 0.0
    
    for tid in cited_ids:
        try:
            # Try both collections
            doc = expense_coll.find_one({'_id': ObjectId(tid)})
            if not doc:
                doc = income_coll.find_one({'_id': ObjectId(tid)})
            
            if doc and doc.get('user_id') == user_id:
                amount = float(doc.get('amount', 0))
                total += amount
        except Exception as e:
            print(f"Error verifying transaction {tid}: {e}")
            continue
    
    return total

def extract_citations_from_answer(answer_text: str) -> List[str]:
    """Extract transaction IDs mentioned in the answer"""
    # Find patterns like [TX_507f1f77bcf86cd799439011] or TX_507f1f77bcf86cd799439011
    pattern = r'(?:\[)?TX_([a-f0-9]{24})(?:\])?'
    citations = re.findall(pattern, answer_text, re.IGNORECASE)
    return citations

def generate_gemini_answer(query: str, context: str) -> str:
    """Generate answer using Gemini Flash"""
    if not model:
        return "(Gemini API not configured) I found relevant transactions based on your query. Please check the verified totals and citations."
    
    prompt = f"""You are a personal finance assistant analyzing transaction data. Based on the provided transactions, answer the user's query with specific details and insights.

IMPORTANT INSTRUCTIONS:
1. Reference specific transactions using the format [TX_<id>] when mentioning amounts or details
2. Provide concrete numbers and calculations
3. Give actionable financial insights
4. Be concise but comprehensive
5. If calculating totals, show your work
6. Identify spending patterns or trends when relevant

USER QUERY: {query}

AVAILABLE TRANSACTIONS:
{context}

Please provide a detailed, helpful response that directly answers the user's question using the transaction data provided."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"I found relevant transactions for your query. Please check the verified totals below. (API Error: {str(e)})"

def generate_answer(user_id: str, query: str, candidates: List[Dict], want_plot: bool = False) -> Dict:
    """Generate comprehensive answer using Gemini Flash with verification"""
    
    # Build context from candidates
    context = build_context_passages(candidates)
    
    # Generate answer using Gemini
    answer_text = generate_gemini_answer(query, context)
    
    # Extract citations from the generated answer
    ai_citations = extract_citations_from_answer(answer_text)
    
    # Use all candidate IDs as fallback citations
    all_candidate_ids = [c['id'] for c in candidates]
    
    # Combine AI-extracted citations with candidate IDs
    final_citations = list(set(ai_citations + all_candidate_ids))
    
    # Verify totals from database
    verified_total = verify_totals_from_db(user_id, final_citations)
    
    # Add summary stats to answer if relevant
    if candidates:
        summary_info = f"\n\n📊 Summary: Found {len(candidates)} relevant transactions totaling ₹{verified_total:,.2f}"
        answer_text += summary_info
    
    return {
        'answer_text': answer_text,
        'citations': final_citations,
        'verified_total': verified_total,
        'plot_path': None  # Will be set by main.py if plot requested
    }