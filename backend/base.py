
from langchain_core.messages import BaseMessage


from typing_extensions import Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()


class LegalAgent(TypedDict):
    messages: Annotated[Sequence[BaseMessage] , add_messages]
    retriever_docs: list[Document]
    tools_used: list[str]
    