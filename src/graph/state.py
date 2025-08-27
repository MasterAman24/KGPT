from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict, total=False):
    user_input: str
    language: str
    english_input: str

    # Planning & execution
    need_tool: bool
    tools_to_call: List[Dict[str, Any]]   
    tool_results: List[Dict[str, Any]]    

    # Final
    final_answer: str
