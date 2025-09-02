from fastapi import FastAPI, HTTPException, UploadFile, File
from app.models.schemas import AddTxRequest, QueryRequest, FeedbackRequest
from app.agents.ingest_agent import ingest_transaction
from app.agents.retriever_agent import hybrid_retrieve
from app.agents.reranker_agent import rerank
from app.agents.fusion_agent import generate_answer
from app.db import mongodb
from app.utils.plotting import plot_spend_by_category, plot_time_series
from fastapi.staticfiles import StaticFiles
import pandas as pd
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Agentic Finance API')

# serve plots
app.mount('/plots', StaticFiles(directory='plots'), name='plots')

@app.post('/transactions/add')
async def add_transaction(req: AddTxRequest):
    try:
        res = await ingest_transaction(req.user_id, req.text)
        return {'status': 'ok', 'result': res}
    except Exception as e:
        logger.error(f"Error adding transaction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to add transaction: {str(e)}")

@app.post('/query')
async def query(req: QueryRequest):
    try:
        logger.info(f"Processing query: {req.query} for user: {req.user_id}")
        
        # 1. retrieve
        logger.info("Step 1: Retrieving candidates")
        candidates = hybrid_retrieve(req.user_id, req.query)
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # 2. rerank
        logger.info("Step 2: Reranking candidates")
        top = rerank(req.query, candidates, top_k=5)
        logger.info(f"Reranked to top {len(top)} candidates")
        
        # 3. generate
        logger.info("Step 3: Generating answer")
        resp = generate_answer(req.user_id, req.query, top, want_plot=req.want_plot)
        logger.info("Answer generated successfully")

        # if user wants plot, build dataframe and plot
        plot_url = None
        if req.want_plot:
            try:
                logger.info("Generating plot")
                # fetch transactions for user and create simple df
                docs = list(mongodb.expense_coll.find({'user_id': req.user_id}))
                logger.info(f"Found {len(docs)} expense documents for plotting")
                
                if docs:
                    df = pd.DataFrame([{
                        'amount': d.get('amount', 0), 
                        'category': d.get('category', 'uncategorized'), 
                        'date': d.get('date')
                    } for d in docs])
                    
                    p = plot_spend_by_category(df, req.user_id)
                    plot_url = f'/plots/{p.split("/")[-1]}'
                    resp['plot_path'] = plot_url
                    logger.info(f"Plot generated: {plot_url}")
                else:
                    logger.warning("No expense documents found for plotting")
            except Exception as plot_error:
                logger.error(f"Plot generation failed: {str(plot_error)}")
                # Don't fail the whole request for plot issues
                resp['plot_error'] = f"Failed to generate plot: {str(plot_error)}"

        return resp
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post('/feedback')
async def feedback(req: FeedbackRequest):
    try:
        mongodb.feedback_coll.insert_one({
            'user_id': req.user_id, 
            'query_id': req.query_id, 
            'thumbs_up': req.thumbs_up
        })
        return {'status': 'saved'}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# Add a health check endpoint
@app.get('/health')
async def health_check():
    try:
        # Test MongoDB connection
        mongodb.income_coll.count_documents({})
        
        # Test ChromaDB connection
        from app.vectorstore.chroma_store import collection
        collection.count()
        
        return {'status': 'healthy', 'message': 'All systems operational'}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {'status': 'unhealthy', 'error': str(e)}

# Add debug endpoint
@app.get('/debug/data/{user_id}')
async def debug_data(user_id: str):
    try:
        # Count documents in each collection
        income_count = mongodb.income_coll.count_documents({'user_id': user_id})
        expense_count = mongodb.expense_coll.count_documents({'user_id': user_id})
        raw_count = mongodb.transactions_raw.count_documents({'user_id': user_id})
        
        # Test vector store
        from app.vectorstore.chroma_store import collection
        vector_count = collection.count()
        
        return {
            'user_id': user_id,
            'mongodb': {
                'income': income_count,
                'expense': expense_count,
                'raw_transactions': raw_count
            },
            'vectorstore': {
                'total_documents': vector_count
            }
        }
    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))