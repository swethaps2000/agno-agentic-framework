# pipelines/agno_pipeline.py
from typing import Dict
from agents.manager_agent import ManagerAgent
# from pipelines.agno_pipeline import AgnoPipeline  # Avoid circular, but self-ref ok for class

class AgnoPipeline:
    def __init__(self):
        self.manager = ManagerAgent()

    async def run_query(self, user_id: str, query: str, want_plot: bool = False) -> Dict:
        import uuid
        query_id = str(uuid.uuid4())
        result = await self.manager.run_query(user_id, query, want_plot, query_id)
        if 'error' in result:
            result['error'] = result['error']
        return result

    async def ingest(self, user_id: str, text: str) -> Dict:
        return await self.manager.run_ingest(user_id, text)

    def save_feedback(self, user_id: str, query_id: str, thumbs_up: bool) -> Dict:
        return self.manager.run_feedback(user_id, query_id, thumbs_up)