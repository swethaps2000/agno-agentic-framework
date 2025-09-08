# agents/retrieval_agent.py
import os
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from db.db_setup import setup_mongo, setup_chroma
from agents.base_agent import agent
from dotenv import load_dotenv

load_dotenv()

@agent(name="RetrievalAgent")
class RetrievalAgent:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_collection = setup_chroma()
        self.mongo_colls = setup_mongo()
        self._docs_cache = []
        self._bm25 = None
        self._doc_ids = []
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def run(self, user_id: str, query: str) -> List[Dict]:
        return self.hybrid_retrieve(user_id, query, k_dense=50, k_sparse=20)
    
    def hybrid_retrieve(self, user_id: str, query: str, k_dense: int = 50, k_sparse: int = 20) -> List[Dict]:
        dense_results = self._dense_retrieve(query, k_dense)
        
        sparse_results = self._sparse_retrieve(query, k_sparse)
        
        combined = self._merge_retrieval_results(dense_results, sparse_results)
        
        return combined
    
    def _dense_retrieve(self, query: str, k: int) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            results = self.chroma_collection.query(
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
        try:
            self._ensure_bm25()
            if not self._bm25 or not self._docs_cache:
                return []
            
            tokenized_query = word_tokenize(query.lower())
            
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
        if self._bm25 is None or len(self._docs_cache) == 0:
            docs = []
            ids = []
            
            for coll in [self.mongo_colls['income_coll'], self.mongo_colls['expense_coll']]:
                for doc in coll.find({}, {'_id': 1, 'notes': 1}).limit(10000):
                    text = doc.get('notes') or ''
                    docs.append(text)
                    ids.append(str(doc['_id']))
            
            self._docs_cache = docs
            self._doc_ids = ids
            
            if docs:
                tokenized = [word_tokenize(d.lower()) for d in docs]
                self._bm25 = BM25Okapi(tokenized)
    
    def _merge_retrieval_results(self, dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
        combined = []
        seen = set()
        
        for item in dense + sparse:
            if item['id'] not in seen:
                combined.append(item)
                seen.add(item['id'])
        
        return combined