from db_connection import retriever, model
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain.agents import create_agent 
from drafter import generate_draft
# from functionalities import tools



@tool("retriever_store")
def retriever_store(query ):
    """Use this tool to retrieve information (before 2025) about supreme court judgements from the vector store or law related sources."""

    print("Retriever Tool Called with query:", query)

    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool("wikipedia_search")
def wikipedia_search(query):
    """Use this tool to get a summary of information from Wikipedia if you didn't get enough information."""
    print("-"*8, "Invoking Wikipedia Search" , "-"*8)
    print("Wikipedia Search Tool Called with query:", query)
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(doc_content_chars_max=1000, top_k_results=1))
    return wiki.invoke(query)


@tool("websearch")
def websearch(query):

    """Use this tool to search the web for the latest information on a topic."""
    print("-"*8, "Invoking Websearch" , "-"*8)
    print("Web Search Tool Called with query:", query)
    tool = TavilySearch(
        max_results=5,
        topic='general',
    )
    
    return tool.invoke({'query': query})


tools = [retriever_store, wikipedia_search , websearch , generate_draft]


react_node = create_agent(model=model, tools=tools)

