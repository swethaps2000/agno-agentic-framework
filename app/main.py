# from fastapi import FastAPI, HTTPException, UploadFile, File
# from app.models.schemas import AddTxRequest, QueryRequest, FeedbackRequest
# from app.agents.ingest_agent import ingest_transaction
# from app.agents.retriever_agent import hybrid_retrieve
# from app.agents.reranker_agent import rerank
# from app.agents.fusion_agent import generate_answer
# from app.db import mongodb
# from app.utils.plotting import generate_smart_plot  # Updated import
# from fastapi.staticfiles import StaticFiles
# import pandas as pd
# import logging
# import traceback

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title='Agentic Finance API')

# # serve plots
# app.mount('/plots', StaticFiles(directory='plots'), name='plots')

# @app.post('/transactions/add')
# async def add_transaction(req: AddTxRequest):
#     try:
#         res = await ingest_transaction(req.user_id, req.text)
#         return {'status': 'ok', 'result': res}
#     except Exception as e:
#         logger.error(f"Error adding transaction: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Failed to add transaction: {str(e)}")

# @app.post('/query')
# async def query(req: QueryRequest):
#     try:
#         logger.info(f"Processing query: {req.query} for user: {req.user_id}")
        
#         # 1. retrieve
#         logger.info("Step 1: Retrieving candidates")
#         candidates = hybrid_retrieve(req.user_id, req.query)
#         logger.info(f"Retrieved {len(candidates)} candidates")
        
#         # 2. rerank
#         logger.info("Step 2: Reranking candidates")
#         top = rerank(req.query, candidates, top_k=5)
#         logger.info(f"Reranked to top {len(top)} candidates")
        
#         # 3. generate
#         logger.info("Step 3: Generating answer")
#         resp = generate_answer(req.user_id, req.query, top, want_plot=req.want_plot)
#         logger.info("Answer generated successfully")

#         # Enhanced plotting based on query analysis
#         plot_url = None
#         plot_description = None
#         if req.want_plot:
#             try:
#                 logger.info("Generating smart plot based on query")
                
#                 # Fetch both expense and income data for comprehensive plotting
#                 expense_docs = list(mongodb.expense_coll.find({'user_id': req.user_id}))
#                 income_docs = list(mongodb.income_coll.find({'user_id': req.user_id}))
                
#                 logger.info(f"Found {len(expense_docs)} expense and {len(income_docs)} income documents")
                
#                 if expense_docs or income_docs:
#                     # Use the new smart plotting function
#                     plot_path, plot_desc = generate_smart_plot(
#                         user_id=req.user_id, 
#                         query=req.query, 
#                         expense_docs=expense_docs,
#                         income_docs=income_docs
#                     )
                    
#                     plot_url = f'/plots/{plot_path.split("/")[-1]}'
#                     plot_description = plot_desc
#                     resp['plot_path'] = plot_url
#                     resp['plot_description'] = plot_description
#                     logger.info(f"Smart plot generated: {plot_url} - {plot_desc}")
#                 else:
#                     logger.warning("No transaction documents found for plotting")
#                     resp['plot_error'] = "No transaction data available for plotting"
                    
#             except Exception as plot_error:
#                 logger.error(f"Plot generation failed: {str(plot_error)}")
#                 logger.error(traceback.format_exc())
#                 # Don't fail the whole request for plot issues
#                 resp['plot_error'] = f"Failed to generate plot: {str(plot_error)}"

#         return resp
        
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# @app.post('/feedback')
# async def feedback(req: FeedbackRequest):
#     try:
#         mongodb.feedback_coll.insert_one({
#             'user_id': req.user_id, 
#             'query_id': req.query_id, 
#             'thumbs_up': req.thumbs_up
#         })
#         return {'status': 'saved'}
#     except Exception as e:
#         logger.error(f"Error saving feedback: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# # Add a health check endpoint
# # @app.get('/health')
# # async def health_check():
# #     try:
# #         # Test MongoDB connection
# #         mongodb.income_coll.count_documents({})
        
# #         # Test ChromaDB connection
# #         from app.vectorstore.chroma_store import collection
# #         collection.count()
        
# #         return {'status': 'healthy', 'message': 'All systems operational'}
# #     except Exception as e:
# #         logger.error(f"Health check failed: {str(e)}")
# #         return {'status': 'unhealthy', 'error': str(e)}

# # # Add debug endpoint
# # @app.get('/debug/data/{user_id}')
# # async def debug_data(user_id: str):
# #     try:
# #         # Count documents in each collection
# #         income_count = mongodb.income_coll.count_documents({'user_id': user_id})
# #         expense_count = mongodb.expense_coll.count_documents({'user_id': user_id})
# #         raw_count = mongodb.transactions_raw.count_documents({'user_id': user_id})
        
# #         # Test vector store
# #         from app.vectorstore.chroma_store import collection
# #         vector_count = collection.count()
        
# #         return {
# #             'user_id': user_id,
# #             'mongodb': {
# #                 'income': income_count,
# #                 'expense': expense_count,
# #                 'raw_transactions': raw_count
# #             },
# #             'vectorstore': {
# #                 'total_documents': vector_count
# #             }
# #         }
# #     except Exception as e:
# #         logger.error(f"Debug endpoint failed: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# # # New endpoint to test plotting functionality
# # @app.get('/debug/plot/{user_id}')
# # async def debug_plot(user_id: str, query: str = "show my spending pattern"):
# #     try:
# #         expense_docs = list(mongodb.expense_coll.find({'user_id': user_id}))
# #         income_docs = list(mongodb.income_coll.find({'user_id': user_id}))
        
# #         plot_path, plot_desc = generate_smart_plot(
# #             user_id=user_id, 
# #             query=query, 
# #             expense_docs=expense_docs,
# #             income_docs=income_docs
# #         )
        
# #         plot_url = f'/plots/{plot_path.split("/")[-1]}'
        
# #         return {
# #             'query': query,
# #             'plot_url': plot_url,
# #             'plot_description': plot_desc,
# #             'data_counts': {
# #                 'expenses': len(expense_docs),
# #                 'income': len(income_docs)
# #             }
# #         }
# #     except Exception as e:
# #         logger.error(f"Debug plot failed: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))
# //////////////////////////////////////////////////////////////////////////////////////////////////////////

"""
Agno Finance Pipeline - Unified Financial Transaction Processing
Converts from multi-agent framework to single Agno pipeline
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Core Dependencies
import google.generativeai as genai
from pymongo import MongoClient
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment
load_dotenv()


class ProcessingStage(Enum):
    """Pipeline stages for transaction processing"""
    INGEST = "ingest"
    PARSE = "parse"
    STORE = "store"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    PLOT = "plot"


@dataclass
class Transaction:
    """Core transaction data structure"""
    id: Optional[str] = None
    user_id: str = ""
    amount: float = 0.0
    type: str = "expense"  # income/expense
    date: str = ""
    category: str = "uncategorized"
    description: str = ""
    notes: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryContext:
    """Query processing context"""
    user_id: str
    query: str
    want_plot: bool = False
    transactions: List[Transaction] = None
    candidates: List[Dict] = None
    answer: str = ""
    plot_path: Optional[str] = None
    plot_description: Optional[str] = None
    verified_total: float = 0.0
    citations: List[str] = None
    
    def __post_init__(self):
        if self.transactions is None:
            self.transactions = []
        if self.candidates is None:
            self.candidates = []
        if self.citations is None:
            self.citations = []


class AgnoFinancePipeline:
    """
    Unified Agno Pipeline for Financial Transaction Processing
    Consolidates: Ingest, Parse, Store, Retrieve, Rerank, Generate, Plot
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self._setup_environment()
        self._setup_database()
        self._setup_vectorstore()
        self._setup_models()
        self._setup_retrieval()
        self._setup_plotting()
        
    def _setup_environment(self):
        """Setup environment variables and configs"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.mongodb_db = os.getenv('MONGODB_DB', 'agentic_finance')
        self.chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './.chromadb')
        self.plots_dir = './plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def _setup_database(self):
        """Setup MongoDB connections"""
        self.mongo_client = MongoClient(self.mongodb_uri)
        self.db = self.mongo_client[self.mongodb_db]
        
        # Collections
        self.income_coll = self.db['income']
        self.expense_coll = self.db['expense']
        self.transactions_raw = self.db['transactions_raw']
        self.feedback_coll = self.db['feedback']
        self.metadata_coll = self.db['metadata']
        
    def _setup_vectorstore(self):
        """Setup Chroma vector store"""
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        try:
            self.collection = self.chroma_client.get_collection("transactions")
        except:
            self.collection = self.chroma_client.create_collection("transactions")
            
    def _setup_models(self):
        """Setup ML models"""
        # Gemini for parsing and generation
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
            
        # Sentence transformers for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def _setup_retrieval(self):
        """Setup retrieval components"""
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # BM25 cache
        self._docs_cache = []
        self._bm25 = None
        self._doc_ids = []
        
    def _setup_plotting(self):
        """Setup plotting configuration"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    # ================== STAGE 1: INGEST & PARSE ==================
    
    async def ingest_transaction(self, user_id: str, text: str) -> Dict:
        """
        Stage 1: Ingest and parse transaction text
        Combines ingest_agent and parsing logic
        """
        # Store raw transaction
        raw_record = {
            "user_id": user_id,
            "text": text,
            "created_at": datetime.utcnow()
        }
        raw_id = self.transactions_raw.insert_one(raw_record).inserted_id
        
        # Parse transaction
        parsed = self._parse_transaction(text)
        
        # Create transaction object
        transaction = Transaction(
            user_id=user_id,
            amount=float(parsed.get('amount', 0.0)),
            type=parsed.get('type', 'expense'),
            date=parsed.get('date', datetime.now().strftime('%Y-%m-%d')),
            category=parsed.get('category', 'uncategorized'),
            description=parsed.get('description', text),
            notes=text
        )
        
        # Store in database
        tx_id = await self._store_transaction(transaction)
        
        # Store in vector database
        await self._vectorize_transaction(tx_id, transaction)
        
        return {
            'raw_id': str(raw_id),
            'tx_id': tx_id,
            'parsed': parsed,
            'transaction': transaction
        }
    
    def _parse_transaction(self, text: str) -> Dict:
        """Parse transaction using Gemini or fallback heuristics"""
        if self.gemini_model:
            return self._gemini_parse(text)
        else:
            return self._heuristic_parse(text)
    
    def _gemini_parse(self, text: str) -> Dict:
        """Use Gemini for transaction parsing"""
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
            
            # Clean markdown
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text.rsplit('\n', 1)[0]
            
            parsed = json.loads(response_text)
            
            # Validate and set defaults
            required_fields = ['amount', 'currency', 'date', 'category', 'type', 'description']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            # Ensure amount is numeric
            if isinstance(parsed['amount'], str):
                parsed['amount'] = float(re.sub(r'[^\d.]', '', parsed['amount']))
                
            return parsed
            
        except Exception as e:
            print(f"Gemini parsing failed: {e}. Using heuristics.")
            return self._heuristic_parse(text)
    
    def _heuristic_parse(self, text: str) -> Dict:
        """Fallback heuristic parsing"""
        # Extract amount
        amount = 0.0
        amount_match = re.search(r'[₹Rs\s]*(\d+[\,\d]*)', text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', ''))
        
        # Detect transaction type
        income_keywords = ['received', 'salary', 'got', 'credited', 'income', 'payment', 'bonus']
        tx_type = 'income' if any(w in text.lower() for w in income_keywords) else 'expense'
        
        # Detect category
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
        """Default values for missing fields"""
        defaults = {
            'amount': 0.0,
            'currency': 'INR',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': 'uncategorized',
            'type': 'expense',
            'description': 'Transaction'
        }
        return defaults.get(field, '')

    # ================== STAGE 2: STORAGE ==================
    
    async def _store_transaction(self, transaction: Transaction) -> str:
        """Store transaction in appropriate MongoDB collection"""
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
            result = self.income_coll.insert_one(record)
        else:
            result = self.expense_coll.insert_one(record)
            
        return str(result.inserted_id)
    
    async def _vectorize_transaction(self, tx_id: str, transaction: Transaction):
        """Store transaction in vector database for semantic search"""
        # Enhanced text for better search
        enhanced_text = f"{transaction.description} | Amount: ₹{transaction.amount} | Type: {transaction.type} | Category: {transaction.category} | Date: {transaction.date}"
        
        # Clean metadata
        metadata = {
            'user_id': transaction.user_id,
            'type': transaction.type,
            'category': transaction.category,
            'amount': float(transaction.amount),
            'date': transaction.date
        }
        
        # Generate embedding
        embedding = self.embedding_model.encode([enhanced_text])[0]
        
        # Store in Chroma
        self.collection.upsert(
            ids=[tx_id],
            metadatas=[metadata],
            documents=[enhanced_text],
            embeddings=[embedding.tolist()]
        )

    # ================== STAGE 3: RETRIEVAL ==================
    
    def hybrid_retrieve(self, user_id: str, query: str, k_dense: int = 50, k_sparse: int = 20) -> List[Dict]:
        """
        Stage 3: Hybrid retrieval combining dense (semantic) and sparse (keyword) search
        """
        # Dense retrieval using embeddings
        dense_results = self._dense_retrieve(query, k_dense)
        
        # Sparse retrieval using BM25
        sparse_results = self._sparse_retrieve(query, k_sparse)
        
        # Combine and deduplicate
        combined = self._merge_retrieval_results(dense_results, sparse_results)
        
        return combined
    
    def _dense_retrieve(self, query: str, k: int) -> List[Dict]:
        """Dense semantic retrieval using Chroma"""
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            items = []
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                items.append({
                    'id': results['ids'][0][i],
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'source': 'dense'
                })
            return items
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            return []
    
    def _sparse_retrieve(self, query: str, k: int) -> List[Dict]:
        """Sparse keyword retrieval using BM25"""
        try:
            self._ensure_bm25()
            if not self._bm25 or not self._docs_cache:
                return []
            
            try:
                tokenized_query = word_tokenize(query.lower())
            except:
                tokenized_query = query.lower().split()
            
            top_docs = self._bm25.get_top_n(tokenized_query, self._docs_cache, n=k)
            
            items = []
            for txt in top_docs:
                try:
                    idx = self._docs_cache.index(txt)
                    items.append({
                        'id': self._doc_ids[idx],
                        'text': txt,
                        'metadata': {},
                        'source': 'sparse'
                    })
                except ValueError:
                    continue
            
            return items
        except Exception as e:
            print(f"Sparse retrieval failed: {e}")
            return []
    
    def _ensure_bm25(self):
        """Initialize BM25 index if needed"""
        if self._bm25 is None or len(self._docs_cache) == 0:
            docs = []
            ids = []
            
            for coll in [self.income_coll, self.expense_coll]:
                for doc in coll.find({}, {'_id': 1, 'notes': 1}).limit(10000):
                    text = doc.get('notes') or ''
                    docs.append(text)
                    ids.append(str(doc['_id']))
            
            self._docs_cache = docs
            self._doc_ids = ids
            
            if docs:
                try:
                    tokenized = [word_tokenize(d.lower()) for d in docs]
                    self._bm25 = BM25Okapi(tokenized)
                except:
                    tokenized = [d.lower().split() for d in docs]
                    self._bm25 = BM25Okapi(tokenized)
    
    def _merge_retrieval_results(self, dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
        """Merge and deduplicate retrieval results"""
        combined = []
        seen = set()
        
        # Prioritize dense results, then add sparse
        for item in dense + sparse:
            if item['id'] not in seen:
                combined.append(item)
                seen.add(item['id'])
        
        return combined

    # ================== STAGE 4: RERANKING ==================
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Stage 4: Rerank candidates using cross-encoder
        """
        if not candidates:
            return []
        
        try:
            # Create query-document pairs
            pairs = [(query, c['text']) for c in candidates]
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Add scores to candidates
            for candidate, score in zip(candidates, scores):
                candidate['score'] = float(score)
            
            # Sort by score and return top-k
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:top_k]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return candidates[:top_k]

    # ================== STAGE 5: ANSWER GENERATION ==================
    
    def generate_answer(self, context: QueryContext) -> QueryContext:
        """
        Stage 5: Generate comprehensive answer using Gemini
        """
        # Build context from candidates
        context_text = self._build_context_passages(context.candidates)
        
        # Generate answer
        answer_text = self._generate_gemini_answer(context.query, context_text)
        
        # Extract citations
        citations = self._extract_citations(answer_text)
        
        # Add fallback citations
        all_candidate_ids = [c['id'] for c in context.candidates]
        final_citations = list(set(citations + all_candidate_ids))
        
        # Verify totals from database
        verified_total = self._verify_totals_from_db(context.user_id, final_citations)
        
        # Add summary
        if context.candidates:
            summary = f"\n\n📊 Summary: Found {len(context.candidates)} relevant transactions totaling ₹{verified_total:,.2f}"
            answer_text += summary
        
        # Update context
        context.answer = answer_text
        context.citations = final_citations
        context.verified_total = verified_total
        
        return context
    
    def _build_context_passages(self, candidates: List[Dict]) -> str:
        """Build context from retrieved candidates"""
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
    
    def _generate_gemini_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini"""
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
        """Extract transaction IDs from answer"""
        pattern = r'(?:\[)?TX_([a-f0-9]{24})(?:\])?'
        return re.findall(pattern, answer_text, re.IGNORECASE)
    
    def _verify_totals_from_db(self, user_id: str, cited_ids: List[str]) -> float:
        """Verify totals by querying MongoDB directly"""
        from bson import ObjectId
        total = 0.0
        
        for tx_id in cited_ids:
            try:
                # Check both collections
                doc = self.expense_coll.find_one({'_id': ObjectId(tx_id)})
                if not doc:
                    doc = self.income_coll.find_one({'_id': ObjectId(tx_id)})
                
                if doc and doc.get('user_id') == user_id:
                    amount = float(doc.get('amount', 0))
                    total += amount
            except Exception as e:
                print(f"Error verifying transaction {tx_id}: {e}")
                continue
        
        return total

    # ================== STAGE 6: PLOTTING ==================
    
    def generate_plot(self, context: QueryContext) -> QueryContext:
        """
        Stage 6: Generate appropriate plot based on query analysis
        """
        if not context.want_plot:
            return context
        
        try:
            # Fetch data for plotting
            expense_docs = list(self.expense_coll.find({'user_id': context.user_id}))
            income_docs = list(self.income_coll.find({'user_id': context.user_id}))
            
            if not expense_docs and not income_docs:
                context.plot_path = None
                context.plot_description = "No transaction data available for plotting"
                return context
            
            # Analyze query for plot type
            plot_analysis = self._analyze_query_for_plot(context.query)
            
            # Generate plot
            plot_path, plot_desc = self._create_smart_plot(
                context.user_id, 
                context.query, 
                expense_docs, 
                income_docs, 
                plot_analysis
            )
            
            context.plot_path = f'/plots/{plot_path.split("/")[-1]}'
            context.plot_description = plot_desc
            
        except Exception as e:
            print(f"Plot generation failed: {e}")
            context.plot_path = None
            context.plot_description = f"Plot generation failed: {str(e)}"
        
        return context
    
    def _analyze_query_for_plot(self, query: str) -> Dict:
        """Analyze query to determine plot type and parameters"""
        query_lower = query.lower()
        
        # Time patterns
        time_patterns = {
            'monthly': r'\b(month|monthly|last\s+\d+\s+months?)\b',
            'weekly': r'\b(week|weekly|last\s+\d+\s+weeks?)\b',
            'yearly': r'\b(year|yearly|annual|last\s+\d+\s+years?)\b',
            'daily': r'\b(day|daily|last\s+\d+\s+days?)\b'
        }
        
        time_period = None
        time_value = None
        for period, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                time_period = period
                number_match = re.search(r'(\d+)\s+' + period.rstrip('ly'), query_lower)
                if number_match:
                    time_value = int(number_match.group(1))
                break
        
        # Plot type detection
        plot_type = 'category_bar'  # default
        if any(word in query_lower for word in ['trend', 'time', 'over time', 'pattern']):
            plot_type = 'time_series'
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage']):
            plot_type = 'pie_chart'
        elif any(word in query_lower for word in ['income', 'expense', 'balance']):
            plot_type = 'income_vs_expense'
        
        # Category detection
        categories = []
        category_keywords = ['groceries', 'rent', 'fuel', 'food', 'bills', 'shopping', 'transport', 'medical', 'investment', 'salary', 'entertainment']
        for cat in category_keywords:
            if cat in query_lower:
                categories.append(cat)
        
        return {
            'plot_type': plot_type,
            'time_period': time_period,
            'time_value': time_value,
            'categories': categories
        }
    
    def _create_smart_plot(self, user_id: str, query: str, expense_docs: List[Dict], income_docs: List[Dict], analysis: Dict) -> Tuple[str, str]:
        """Create appropriate plot based on analysis"""
        # Convert to DataFrames
        expense_df = pd.DataFrame(expense_docs) if expense_docs else pd.DataFrame()
        income_df = pd.DataFrame(income_docs) if income_docs else pd.DataFrame()
        
        # Filter data
        if not expense_df.empty:
            expense_df = self._filter_data_by_query(expense_df, analysis)
        if not income_df.empty:
            income_df = self._filter_data_by_query(income_df, analysis)
        
        plot_type = analysis['plot_type']
        timestamp = int(datetime.now().timestamp())
        
        try:
            if plot_type == 'time_series':
                path = self._plot_time_series(expense_df, user_id, analysis, timestamp)
                desc = f"Time series showing spending over {analysis.get('time_period', 'time')}"
            elif plot_type == 'pie_chart':
                path = self._plot_pie_chart(expense_df, user_id, timestamp)
                desc = "Pie chart showing spending distribution by category"
            elif plot_type == 'income_vs_expense':
                path = self._plot_income_vs_expense(income_df, expense_df, user_id, timestamp)
                desc = "Income vs expense comparison with net savings/loss"
            else:  # category_bar
                path = self._plot_category_breakdown(expense_df, user_id, analysis, timestamp)
                desc = "Category breakdown of spending"
            
            return path, desc
            
        except Exception as e:
            print(f"Plot creation failed: {e}")
            path = self._create_empty_plot(user_id, f"Error: {str(e)}", timestamp)
            return path, "Error generating plot"
    
    def _filter_data_by_query(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Filter data based on query analysis"""
        if df.empty:
            return df
        
        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Time filtering
        if analysis['time_period'] and analysis['time_value']:
            end_date = datetime.now()
            if analysis['time_period'] == 'monthly':
                start_date = end_date - timedelta(days=30 * analysis['time_value'])
            elif analysis['time_period'] == 'weekly':
                start_date = end_date - timedelta(weeks=analysis['time_value'])
            elif analysis['time_period'] == 'yearly':
                start_date = end_date - timedelta(days=365 * analysis['time_value'])
            else:  # daily
                start_date = end_date - timedelta(days=analysis['time_value'])
            
            df = df[df['date'] >= start_date]
        
        # Category filtering
        if analysis['categories'] and 'category' in df.columns:
            df = df[df['category'].isin(analysis['categories'])]
        
        return df
    
    def _plot_category_breakdown(self, df: pd.DataFrame, user_id: str, analysis: Dict, timestamp: int) -> str:
        """Create category breakdown bar chart"""
        if df.empty:
            return self._create_empty_plot(user_id, "No expense data available", timestamp)
        
        agg = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(agg.index, agg.values, color=sns.color_palette("husl", len(agg)))
        
        ax.set_title('Spending by Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amount (₹)', fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'₹{height:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_category_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_time_series(self, df: pd.DataFrame, user_id: str, analysis: Dict, timestamp: int) -> str:
        """Create time series plot"""
        if df.empty:
            return self._create_empty_plot(user_id, "No data for time series", timestamp)
        
        # Group by time period
        period_map = {
            'monthly': 'M',
            'weekly': 'W',
            'yearly': 'Y',
            'daily': 'D'
        }
        period = period_map.get(analysis.get('time_period'), 'D')
        
        df['period'] = df['date'].dt.to_period(period)
        grouped = df.groupby('period')['amount'].sum().reset_index()
        grouped['period_str'] = grouped['period'].astype(str)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(grouped['period_str'], grouped['amount'], marker='o', linewidth=2, markersize=6)
        
        ax.set_title(f'{analysis.get("time_period", "Daily").title()} Spending Pattern', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Amount (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_timeseries_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_pie_chart(self, df: pd.DataFrame, user_id: str, timestamp: int) -> str:
        """Create pie chart for category distribution"""
        if df.empty:
            return self._create_empty_plot(user_id, "No data for pie chart", timestamp)
        
        agg = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Top 8 categories, rest as "Others"
        if len(agg) > 8:
            top_categories = agg.head(8)
            others_sum = agg.tail(len(agg) - 8).sum()
            if others_sum > 0:
                top_categories['Others'] = others_sum
            agg = top_categories
        
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = sns.color_palette("husl", len(agg))
        
        wedges, texts, autotexts = ax.pie(agg.values, labels=agg.index, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Spending Distribution by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, f'{user_id}_pie_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _plot_income_vs_expense(self, income_df: pd.DataFrame, expense_df: pd.DataFrame, 
                               user_id: str, timestamp: int) -> str:
        """Create income vs expense comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Monthly aggregation
        if not expense_df.empty:
            expense_df['month'] = pd.to_datetime(expense_df['date']).dt.to_period('M')
            expense_monthly = expense_df.groupby('month')['amount'].sum()
        else:
            expense_monthly = pd.Series(dtype=float)
        
        if not income_df.empty:
            income_df['month'] = pd.to_datetime(income_df['date']).dt.to_period('M')
            income_monthly = income_df.groupby('month')['amount'].sum()
        else:
            income_monthly = pd.Series(dtype=float)
        
        # Get all months
        all_months = sorted(set(list(expense_monthly.index) + list(income_monthly.index)))
        
        if all_months:
            income_values = [income_monthly.get(month, 0) for month in all_months]
            expense_values = [expense_monthly.get(month, 0) for month in all_months]
            month_labels = [str(month) for month in all_months]
            
            # Bar comparison
            x = np.arange(len(month_labels))
            width = 0.35
            
            ax1.bar(x - width/2, income_values, width, label='Income', color='green', alpha=0.7)
            ax1.bar(x + width/2, expense_values, width, label='Expense', color='red', alpha=0.7)
            
            ax1.set_title('Monthly Income vs Expense', fontweight='bold')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Amount (₹)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(month_labels, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Net savings/loss
            net_values = [inc - exp for inc, exp in zip(income_values, expense_values)]
            colors = ['green' if x >= 0 else 'red' for x in net_values]
            
            ax2.bar(month_labels, net_values, color=colors, alpha=0.7)
            ax2.set_title('Monthly Savings/Loss', fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Net Amount (₹)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f'{user_id}_income_expense_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path
    
    def _create_empty_plot(self, user_id: str, message: str, timestamp: int) -> str:
        """Create empty plot with message"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16,
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('No Data Available', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        path = os.path.join(self.plots_dir, f'{user_id}_empty_{timestamp}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path

    # ================== MAIN PIPELINE ORCHESTRATOR ==================
    
    async def process_query(self, user_id: str, query: str, want_plot: bool = False) -> Dict:
        """
        Main pipeline orchestrator - processes complete query through all stages
        """
        print(f"🚀 Starting Agno Pipeline for user {user_id}: {query}")
        
        # Initialize context
        context = QueryContext(
            user_id=user_id,
            query=query,
            want_plot=want_plot
        )
        
        try:
            # Stage 3: Retrieve relevant transactions
            print("📊 Stage 3: Hybrid Retrieval")
            context.candidates = self.hybrid_retrieve(user_id, query)
            print(f"   Retrieved {len(context.candidates)} candidates")
            
            # Stage 4: Rerank candidates
            print("🎯 Stage 4: Reranking")
            context.candidates = self.rerank_candidates(query, context.candidates, top_k=5)
            print(f"   Reranked to top {len(context.candidates)} candidates")
            
            # Stage 5: Generate answer
            print("✍️ Stage 5: Answer Generation")
            context = self.generate_answer(context)
            print(f"   Generated answer with {len(context.citations)} citations")
            
            # Stage 6: Generate plot if requested
            if want_plot:
                print("📈 Stage 6: Plot Generation")
                context = self.generate_plot(context)
                print(f"   Plot status: {context.plot_description}")
            
            # Return results
            result = {
                'answer_text': context.answer,
                'citations': context.citations,
                'verified_total': context.verified_total,
                'plot_path': context.plot_path,
                'plot_description': context.plot_description,
                'pipeline_stages': [
                    ProcessingStage.RETRIEVE.value,
                    ProcessingStage.RERANK.value,
                    ProcessingStage.GENERATE.value,
                ]
            }
            
            if want_plot:
                result['pipeline_stages'].append(ProcessingStage.PLOT.value)
            
            print("✅ Pipeline completed successfully")
            return result
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return {
                'error': f"Pipeline processing failed: {str(e)}",
                'answer_text': "I encountered an error processing your query. Please try again.",
                'citations': [],
                'verified_total': 0.0,
                'plot_path': None,
                'plot_description': None
            }
    
    async def add_transaction(self, user_id: str, text: str) -> Dict:
        """
        Process new transaction through ingest pipeline
        """
        print(f"💰 Processing transaction for user {user_id}: {text}")
        
        try:
            # Stages 1-2: Ingest, Parse, Store, Vectorize
            result = await self.ingest_transaction(user_id, text)
            
            print(f"✅ Transaction processed successfully: {result['tx_id']}")
            return {
                'status': 'success',
                'transaction_id': result['tx_id'],
                'parsed_data': result['parsed'],
                'pipeline_stages': [
                    ProcessingStage.INGEST.value,
                    ProcessingStage.PARSE.value,
                    ProcessingStage.STORE.value
                ]
            }
            
        except Exception as e:
            print(f"❌ Transaction processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': f"Transaction processing failed: {str(e)}"
            }
    
    def save_feedback(self, user_id: str, query_id: str, thumbs_up: bool) -> Dict:
        """Save user feedback"""
        try:
            self.feedback_coll.insert_one({
                'user_id': user_id,
                'query_id': query_id,
                'thumbs_up': thumbs_up,
                'timestamp': datetime.utcnow()
            })
            return {'status': 'saved'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# ================== FASTAPI INTEGRATION ==================

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Request/Response Models
class AddTransactionRequest(BaseModel):
    user_id: str
    text: str

class QueryRequest(BaseModel):
    user_id: str
    query: str
    want_plot: bool = False

class FeedbackRequest(BaseModel):
    user_id: str
    query_id: str
    thumbs_up: bool

# FastAPI App
app = FastAPI(title='Agno Finance Pipeline API')

# Mount static files for plots
app.mount('/plots', StaticFiles(directory='plots'), name='plots')

# Initialize pipeline
pipeline = AgnoFinancePipeline()

@app.post('/transactions/add')
async def add_transaction_endpoint(req: AddTransactionRequest):
    """Add new transaction through Agno pipeline"""
    try:
        result = await pipeline.add_transaction(req.user_id, req.text)
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['error'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction processing failed: {str(e)}")

@app.post('/query')
async def query_endpoint(req: QueryRequest):
    """Process query through complete Agno pipeline"""
    try:
        result = await pipeline.process_query(req.user_id, req.query, req.want_plot)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post('/feedback')
async def feedback_endpoint(req: FeedbackRequest):
    """Save user feedback"""
    try:
        result = pipeline.save_feedback(req.user_id, req.query_id, req.thumbs_up)
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['error'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'pipeline': 'Agno Finance Pipeline',
        'version': '1.0.0',
        'stages': [stage.value for stage in ProcessingStage]
    }

@app.get('/stats/{user_id}')
async def get_user_stats(user_id: str):
    """Get user transaction statistics"""
    try:
        # Get transaction counts
        expense_count = pipeline.expense_coll.count_documents({'user_id': user_id})
        income_count = pipeline.income_coll.count_documents({'user_id': user_id})
        
        # Get total amounts
        expense_pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': None, 'total': {'$sum': '$amount'}}}
        ]
        income_pipeline = expense_pipeline.copy()
        
        expense_total = list(pipeline.expense_coll.aggregate(expense_pipeline))
        income_total = list(pipeline.income_coll.aggregate(income_pipeline))
        
        expense_sum = expense_total[0]['total'] if expense_total else 0
        income_sum = income_total[0]['total'] if income_total else 0
        
        return {
            'user_id': user_id,
            'transactions': {
                'expense_count': expense_count,
                'income_count': income_count,
                'total_count': expense_count + income_count
            },
            'amounts': {
                'total_expenses': expense_sum,
                'total_income': income_sum,
                'net_savings': income_sum - expense_sum
            },
            'pipeline_ready': True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Agno Finance Pipeline")
    print("📋 Available endpoints:")
    print("   POST /transactions/add - Add new transaction")
    print("   POST /query - Process financial query")
    print("   POST /feedback - Save user feedback")
    print("   GET /health - Health check")
    print("   GET /stats/{user_id} - User statistics")
    uvicorn.run(app, host="0.0.0.0", port=8000)