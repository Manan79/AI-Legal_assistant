
from langchain_core.messages import BaseMessage


from typing_extensions import Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"


os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

os.environ["LANGCHAIN_PROJECT"]="Legal_AI_Assistant"

class LegalAgent(TypedDict):
    messages: Annotated[Sequence[BaseMessage] , add_messages]
    retriever_docs: list[Document]
    tools_used: list[str]
    draft: str
    draft_type: str

    
