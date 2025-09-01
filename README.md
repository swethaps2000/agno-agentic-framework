This repository provides a working scaffold for an agentic AI that ingests natural-language transactions,
stores them in MongoDB (separate collections for `income` and `expense`), indexes transaction text in Chroma,
performs hybrid retrieval (BM25 + dense), re-ranks using a Cross-Encoder, and generates answers via Gemini.

Files included:
- app/main.py
- app/agents/ingest_agent.py
- app/agents/retriever_agent.py
- app/agents/reranker_agent.py
- app/agents/fusion_agent.py
- app/db/mongodb.py
- app/vectorstore/chroma_store.py
- app/utils/plotting.py
- app/models/schemas.py
- requirements.txt


Environment variables (create a .env file):
- MONGODB_URI
- MONGODB_DB
- GEMINI_API_KEY
- CHROMA_PERSIST_DIR (optional)


Run:
1. pip install -r requirements.txt
2. uvicorn app.main:app --reload