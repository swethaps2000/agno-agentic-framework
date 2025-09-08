# db/db_setup.py
import os
from pymongo import MongoClient
import chromadb
from dotenv import load_dotenv

load_dotenv()

def setup_mongo():
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    mongodb_db = os.getenv('MONGODB_DB', 'agentic_finance')
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db]
    return {
        'income_coll': db['income'],
        'expense_coll': db['expense'],
        'transactions_raw': db['transactions_raw'],
        'feedback_coll': db['feedback'],
        'metadata_coll': db['metadata']
    }

def setup_chroma():
    chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './.chromadb')
    client = chromadb.PersistentClient(path=chroma_persist_dir)
    try:
        collection = client.get_collection("transactions")
    except:
        collection = client.create_collection("transactions")
    return collection