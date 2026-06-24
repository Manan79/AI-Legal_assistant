from langgraph.graph import StateGraph , START , END
from backend.base import LegalAgent
from IPython.display import Markdown, display
from langchain.messages import HumanMessage
from backend.functionalities import react_node
from langsmith import traceable


def workflow():
    builder = StateGraph(LegalAgent)

    builder.add_node("React_agent_legal", react_node)

    builder.add_edge(START, "React_agent_legal")
    builder.add_edge("React_agent_legal", END)

    workflow = builder.compile()

    return workflow


new_workflow = workflow()



