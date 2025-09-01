from app.vectorstore.chroma_store import query_dense
from rank_bm25 import BM25Okapi
from app.db.mongodb import transactions_raw, income_coll, expense_coll
from typing import List
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# We'll maintain a small in-memory BM25 index for demonstration. In prod, persist tokens + docs.
_docs_cache = []
_bm25 = None
_doc_ids = []


def _ensure_bm25():
    global _bm25, _docs_cache, _doc_ids
    if _bm25 is None or len(_docs_cache) == 0:
    # load latest from mongo (limit sample)
        docs = []
        ids = []
        for coll in [income_coll, expense_coll]:
            for d in coll.find({}, {'_id':1, 'notes':1}).limit(10000):
                txt = d.get('notes') or ''
                docs.append(txt)
                ids.append(str(d['_id']))
        _docs_cache = docs
        _doc_ids = ids
        tokenized = [word_tokenize(d.lower()) for d in _docs_cache]
        _bm25 = BM25Okapi(tokenized)




def hybrid_retrieve(user_id: str, query: str, k_dense=50, k_sparse=20):
# dense candidates from Chroma
    dense = query_dense(query, k=k_dense)
    dense_ids = {item['id']: item for item in dense}


# sparse via BM25
    _ensure_bm25()
    tokenized_query = word_tokenize(query.lower())
    top_n = _bm25.get_top_n(tokenized_query, _docs_cache, n=k_sparse)
    sparse_items = []
    for txt in top_n:
        # find index -> id
        try:
            idx = _docs_cache.index(txt)
            sparse_items.append({'id': _doc_ids[idx], 'text': txt, 'metadata': {}})
        except ValueError:
            continue


    # merge preserving density ranking, then sparse
    combined = []
    seen = set()
    for item in dense + sparse_items:
        if item['id'] not in seen:
            combined.append(item)
            seen.add(item['id'])
    return combined