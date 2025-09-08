# agents/manager_agent.py
from typing import Dict
from agents.retrieval_agent import RetrievalAgent
from agents.rerank_agent import RerankAgent
from agents.fusion_agent import FusionAgent
from agents.generator_agent import GeneratorAgent
from agents.plot_agent import PlotAgent
from agents.feedback_agent import FeedbackAgent
from agents.ingestion_agent import IngestionAgent
from agents.base_agent import agent

@agent(name="ManagerAgent")
class ManagerAgent:
    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.rerank = RerankAgent()
        self.fusion = FusionAgent()
        self.generator = GeneratorAgent()
        self.plot = PlotAgent()
        self.feedback = FeedbackAgent()

    async def run_query(self, user_id: str, query: str, want_plot: bool, query_id: str) -> Dict:
        candidates = self.retrieval.run(user_id, query)
        reranked = self.rerank.run(query, candidates)
        fused = self.fusion.run(reranked)
        gen_output = self.generator.run(user_id, query, fused)
        plot_output = self.plot.run(user_id, query, want_plot)
        result = {**gen_output, **plot_output}
        # Feedback loop placeholder (actual feedback via endpoint)
        return result

    async def run_ingest(self, user_id: str, text: str) -> Dict:
        ingestion = IngestionAgent()
        return await ingestion.run(user_id, text)

    def run_feedback(self, user_id: str, query_id: str, thumbs_up: bool) -> Dict:
        return self.feedback.run(user_id, query_id, thumbs_up)