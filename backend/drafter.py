
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Markdown, display
from langchain.tools import tool

import markdown
load_dotenv()
model_draft = GoogleGenerativeAI(model = 'gemini-2.5-flash')

SYSTEM = """You are an expert Indian Legal Drafting Assistant.

You can draft: Legal Notices, Affidavits, RTI Applications,
NDAs, Rental Agreements, Consumer Complaints, and MOUs.

RULES:
1. Always use formal Indian legal language.
2. Never invent facts, names, addresses, or dates — use only what the user provides.
3. Use [PLACEHOLDER] for any user detail.
4. Only cite Indian laws/sections you are confident about.
5. List any assumptions separately at the end of the document.
6. keep it short not extra or irrelevant information.
"""

draft_prompt = ChatPromptTemplate.from_messages([

        ("system", SYSTEM),

        ("human",
        """
        Document Type: {draft_type}

        User Request: {query}
        """)

    ])

    


# model = ChatGroq(
#     model_name="openai/gpt-oss-120b",
#     temperature=0)

@tool
def generate_draft(query , draft_type:str):
    """ Use this function for drafting any legal documents Legal Notices, Affidavits, RTI Applications,NDAs, Rental Agreements, Consumer Complaints, and MOUs.
    args:
        user_query: The user query given, draft type: which type of draft needes to generate eg rental agreeement, legal notice etc. 
     """
    try:
        with open(f"backend/drafted_agreements/{draft_type}.txt" , 'r') as f:
            result = f.read()
    except:
        response = draft_prompt | model_draft

        result = response.invoke({
            "draft_type": draft_type ,
            "query":query,
        })

        md_text = f"### Answer\n\n{result}"
        result = markdown.markdown(md_text)

    return result

