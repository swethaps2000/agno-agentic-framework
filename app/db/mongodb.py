import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()


MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DB = os.getenv('MONGODB_DB', 'agentic_finance')


client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]


# collections: income, expense, transactions_raw, feedback, metadata
income_coll = db['income']
expense_coll = db['expense']
transactions_raw = db['transactions_raw']
feedback_coll = db['feedback']
metadata_coll = db['metadata']