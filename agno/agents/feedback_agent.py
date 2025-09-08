# agents/feedback_agent.py
from typing import Dict
from datetime import datetime
from db.db_setup import setup_mongo
from agents.base_agent import agent

@agent(name="FeedbackAgent")
class FeedbackAgent:
    def __init__(self):
        self.feedback_coll = setup_mongo()['feedback_coll']

    def run(self, user_id: str, query_id: str, thumbs_up: bool) -> Dict:
        try:
            self.feedback_coll.insert_one({
                'user_id': user_id,
                'query_id': query_id,
                'thumbs_up': thumbs_up,
                'timestamp': datetime.utcnow()
            })
            return {'status': 'saved'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}