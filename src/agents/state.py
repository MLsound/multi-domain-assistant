from typing import TypedDict, List, Dict, Any, Annotated
import operator

class GraphState(TypedDict):
    query: str
    category_probs: Dict[str, float]
    retrieved_chunks: List[Dict[str, Any]]
    context_metadata: Dict[str, Any]
    response: str
    is_help_section: bool
    # To maintain history if needed, but for now we focus on the flow
    history: Annotated[List[Dict[str, str]], operator.add]
