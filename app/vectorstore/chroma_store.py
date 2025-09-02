import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './.chromadb')

_chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# collection name: transactions
try:
    collection = _chroma_client.get_collection("transactions")
except Exception:
    collection = _chroma_client.create_collection("transactions")

# local SBERT model for embeddings (dense)
_emb_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: List[str]):
    return _emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

async def upsert_transaction(tx_id: str, record: dict, metadata: Dict):
    """
    Upsert transaction - extract text from record and create enhanced text for embedding
    """
    # Extract text from the record
    text = record.get('notes', '')
    
    # Create enhanced text with structured information for better search
    enhanced_text = f"{text} | Amount: ₹{record.get('amount', 0)} | Type: {record.get('type', 'unknown')} | Category: {record.get('category', 'uncategorized')}"
    
    # Clean metadata to ensure JSON serializable values
    clean_metadata = {
        'user_id': metadata.get('user_id', ''),
        'type': metadata.get('type', ''),
        'category': metadata.get('category', ''),
        'amount': float(metadata.get('amount', 0)),
        'date': str(metadata.get('date', ''))
    }
    
    vec = embed_texts([enhanced_text])[0]
    collection.upsert(ids=[tx_id], metadatas=[clean_metadata], documents=[enhanced_text], embeddings=[vec.tolist()])

def query_dense(query: str, k: int = 50):
    qv = embed_texts([query])[0]
    results = collection.query(query_embeddings=[qv.tolist()], n_results=k)
    # results: dict with ids, metadatas, documents, distances
    items = []
    for i, doc in enumerate(results.get('documents', [[]])[0]):
        items.append({
            'id': results['ids'][0][i],
            'text': doc,
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    return items  # Fixed indentation - this was inside the loop!