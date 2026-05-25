from langchain_astradb import AstraDBVectorStore
import os
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from functools import cache

load_dotenv()
os.environ["HF_HOME"]="./hf_cache"

@cache
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache"
    )

embeddings=load_embeddings()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("........ Establishing DB connection ........")

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="AI_Legal_database",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=None,
    autodetect_collection=True,
)

print("------------- Connection Established ---------------")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":10,
        "fetch_k":30,
        "lambda_mult":0.7
    }
)


# print("VectorStore Connected successfully" , vectorstore)