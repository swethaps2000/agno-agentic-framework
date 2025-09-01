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


app = FastAPI(title='Agentic Finance API')


# serve plots
app.mount('/plots', StaticFiles(directory='plots'), name='plots')


@app.post('/transactions/add')
async def add_transaction(req: AddTxRequest):
    res = await ingest_transaction(req.user_id, req.text)
    return {'status':'ok', 'result': res}


@app.post('/query')
async def query(req: QueryRequest):
    # 1. retrieve
    candidates = hybrid_retrieve(req.user_id, req.query)
    # 2. rerank
    top = rerank(req.query, candidates, top_k=5)
    # 3. generate
    resp = generate_answer(req.user_id, req.query, top, want_plot=req.want_plot)


    # if user wants plot, build dataframe and plot
    plot_url = None
    if req.want_plot:
        # fetch transactions for user and create simple df
        docs = list(mongodb.expense_coll.find({'user_id': req.user_id}))
        if docs:
            df = pd.DataFrame([{'amount': d.get('amount',0), 'category': d.get('category','uncategorized'), 'date': d.get('date')} for d in docs])
            p = plot_spend_by_category(df, req.user_id)
            plot_url = f'/plots/{p.split("/")[-1]}'
            resp['plot_path'] = plot_url    


    return resp


@app.post('/feedback')
async def feedback(req: FeedbackRequest):
    mongodb.feedback_coll.insert_one({'user_id': req.user_id, 'query_id': req.query_id, 'thumbs_up': req.thumbs_up})
    return {'status':'saved'}