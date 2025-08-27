
from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import decide_tool_node, multi_tool_node, answer_node

def build_graph():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("decide", decide_tool_node)
    workflow.add_node("multi_tool", multi_tool_node)
    workflow.add_node("answer", answer_node)

    # Route: decide → (multi_tool | answer)
    def route_decision(state: AgentState) -> str:
        return "multi_tool" if state.get("need_tool") else "answer"

    workflow.add_edge(START, "decide")
    workflow.add_conditional_edges("decide", route_decision, {
        "multi_tool": "multi_tool",
        "answer": "answer"
    })
    workflow.add_edge("multi_tool", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()
