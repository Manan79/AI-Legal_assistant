from langgraph.graph import StateGraph , START , END
from typing_extensions import Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from base import LegalAgent
from IPython.display import Markdown, display

from functionalities import react_node


builder = StateGraph(LegalAgent)

builder.add_node("React_agent_legal", react_node)

builder.add_edge(START, "React_agent_legal")
builder.add_edge("React_agent_legal", END)


workflow = builder.compile()

try:
    result = workflow.invoke({"messages": [HumanMessage(content="There are some recent developments in the supreme court judgements related to technology, can you tell me about them?")]})

    display(Markdown(f"### Answer\n\n{result['messages'][-1].content}"))

except Exception as e:
    print(e)

