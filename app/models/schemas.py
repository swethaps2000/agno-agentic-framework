from pydantic import BaseModel
from typing import Optional
from datetime import date


class AddTxRequest(BaseModel):
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