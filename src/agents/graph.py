from langgraph.graph import StateGraph, END
from src.agents.state import GraphState
from src.router.mlp_router import MLPRouter
from src.retrieval.weighted_retriever import WeightedRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

class RAGGraph:
    def __init__(self, mlp_model_path: str = None):
        self.router = MLPRouter(mlp_model_path)
        self.retriever = WeightedRetriever()
        # Initialize Gemini LLM using a confirmed available stable alias
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest") 

        workflow = StateGraph(GraphState)

        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retrieval_node)
        workflow.add_node("synthesis", self.synthesis_node)

        workflow.set_entry_point("router")
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "synthesis")
        workflow.add_edge("synthesis", END)

        self.app = workflow.compile()

    def router_node(self, state: GraphState):
        query = state["query"]
        probs = self.router.route(query)
        return {"category_probs": probs}

    def retrieval_node(self, state: GraphState):
        query = state["query"]
        probs = state["category_probs"]
        is_help = state.get("is_help_section", False)

        # Hardcoded override for "Help" section
        if is_help:
            probs = {"Software": 0.85, "User": 0.0, "Science": 0.15}
        
        chunks = self.retriever.search(query, probs)
        
        # Science-based data enrichment (simple heuristic for now)
        context_metadata = {}
        if probs.get("Science", 0) > 0.4:
            context_metadata = {"Environmental Conditions": "Temperature: 22C, Humidity: 45%"} # Example of adding environmental metadata for science queries
            
        return {"retrieved_chunks": chunks, "context_metadata": context_metadata, "category_probs": probs}

    def synthesis_node(self, state: GraphState):
        chunks = state["retrieved_chunks"]
        context_metadata = state["context_metadata"]
        query = state["query"]
        
        context_str = "\n\n".join([f"[{c['category']}] {c['content']}" for c in chunks])
        if context_metadata:
            context_str += f"\n\n[Environmental Metadata]: {context_metadata}"

        system_prompt = (
            "You are a Senior AI Architect. Synthesize a response based ONLY on the provided context. "
            "Maintain a Truth-Maintenance check: if the information is not in the context, state you don't know. "
            "Ensure the answer is grounded in the retrieved sources."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuery: {query}")
        ]
        
        response = self.llm.invoke(messages)
        return {"response": response.content}
