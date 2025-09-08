# app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipelines.agno_pipeline import AgnoPipeline
import uvicorn
from typing import Optional
from enum import Enum

# Define ProcessingStage for health endpoint
class ProcessingStage(Enum):
    INGEST = "ingest"
    PARSE = "parse"
    STORE = "store"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    PLOT = "plot"

# Initialize FastAPI app
app = FastAPI(title='Agno Finance Pipeline API')

# Mount static files for plots
app.mount('/plots', StaticFiles(directory='plots'), name='plots')

# Initialize pipeline
pipeline = AgnoPipeline()

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

@app.post('/transactions/add')
async def add_transaction_endpoint(req: AddTransactionRequest):
    """Add new transaction through Agno pipeline"""
    try:
        result = await pipeline.ingest(req.user_id, req.text)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction processing failed: {str(e)}")

@app.post('/query')
async def query_endpoint(req: QueryRequest):
    """Process query through complete Agno pipeline"""
    try:
        result = await pipeline.run_query(req.user_id, req.query, req.want_plot)
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        # Inline citations: Append <grok:citation> placeholder (adjust as needed for frontend)
        for cit in result.get('citations', []):
            result['answer'] += f' <grok:{cit}>'
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post('/feedback')
def feedback_endpoint(req: FeedbackRequest):
    """Save user feedback"""
    try:
        result = pipeline.save_feedback(req.user_id, req.query_id, req.thumbs_up)
        if 'error' in result:
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
        from db.db_setup import setup_mongo
        mongo_colls = setup_mongo()
        # Get transaction counts
        expense_count = mongo_colls['expense_coll'].count_documents({'user_id': user_id})
        income_count = mongo_colls['income_coll'].count_documents({'user_id': user_id})
        
        # Get total amounts
        expense_pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': None, 'total': {'$sum': '$amount'}}}
        ]
        income_pipeline = expense_pipeline.copy()
        
        expense_total = list(mongo_colls['expense_coll'].aggregate(expense_pipeline))
        income_total = list(mongo_colls['income_coll'].aggregate(income_pipeline))
        
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
    print("🚀 Starting Agno Finance Pipeline")
    print("📋 Available endpoints:")
    print("   POST /transactions/add - Add new transaction")
    print("   POST /query - Process financial query")
    print("   POST /feedback - Save user feedback")
    print("   GET /health - Health check")
    print("   GET /stats/{user_id} - User statistics")
    uvicorn.run(app, host="0.0.0.0", port=8000)