from langgraph.graph import StateGraph , START , END
from typing_extensions import Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from base import LegalAgent
from IPython.display import Markdown, display

from functionalities import react_node
from langsmith import traceable


def workflow():
    builder = StateGraph(LegalAgent)

    builder.add_node("React_agent_legal", react_node)

    builder.add_edge(START, "React_agent_legal")
    builder.add_edge("React_agent_legal", END)


    workflow = builder.compile()

    return workflow


new_workflow = workflow()

# try:
#     result = new_workflow.invoke({"messages": [HumanMessage(content="Can you generate a timeline for Arvind Kejriwal CBI case with little bit summary ?")]})
#     # display(Markdown(f"### Answer\n\n{result['messages'][-1].content}"))
#     print(result['messages'][-1].content)

# except Exception as e:
#     print(e)

