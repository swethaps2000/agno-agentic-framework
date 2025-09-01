from sentence_transformers import CrossEncoder


# small cross-encoder; may be heavy to load - adjust per infra
_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank(query: str, candidates: list, top_k: int = 5):
    pairs = [(query, c['text']) for c in candidates]
    scores = _reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c['score'] = float(s)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]