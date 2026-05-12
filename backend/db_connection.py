from langchain_astradb import AstraDBVectorStore
import os
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="AI_Legal_database",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=None,
    autodetect_collection=True,
)

retriever = vectorstore.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs={"k": 3 , "score_threshold": 0.35}
)

# print("VectorStore Connected successfully" , vectorstore)