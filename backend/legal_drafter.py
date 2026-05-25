import re
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — Define the shared state
# This dict is passed between every node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class State(TypedDict):
    user_input:     str   # what the user typed
    cot_analysis:   str   # reasoning output from analyse node
    ready:          bool  # True = draft it, False = ask questions
    missing:        str   # comma-separated missing fields
    output:         str   # final response shown to user


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — Set up the LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


llm = GoogleGenerativeAI(model = 'gemini-2.5-flash')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — The system prompt (your legal persona)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM = """You are an expert Indian Legal Drafting Assistant.

You can draft: Legal Notices, Affidavits, RTI Applications, FIR Drafts,
NDAs, Rental Agreements, Consumer Complaints, Petitions, and MOUs.

RULES:
1. Always use formal Indian legal language.
2. Never invent facts, names, addresses, or dates — use only what the user provides.
3. Use [PLACEHOLDER] for any missing detail.
4. Only cite Indian laws/sections you are confident about.
5. List any assumptions separately at the end of the document.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1: analyse
# Reads the user request and decides:
#   - Is enough info present to draft? → READY_TO_DRAFT: YES
#   - What's missing?                  → READY_TO_DRAFT: NO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse(state: State) -> State:
    prompt = f"""{SYSTEM}

        The user wants a legal document. Analyse their request step by step.

        <thinking>
        STEP 1 — What type of document is needed?
        STEP 2 — Who are the parties (sender, recipient)?
        STEP 3 — What critical information is provided vs missing?
                Required: sender name/address, recipient name/address,
                subject matter, key facts, dates, relief sought.
        STEP 4 — Which Indian laws/sections apply?
        STEP 5 — Any stamp duty or notarisation requirements?
        </thinking>

        User Request:
        {state['user_input']}

        After your analysis write EXACTLY one of these on the last line:
        READY_TO_DRAFT: YES
        or
        READY_TO_DRAFT: NO | MISSING: field1, field2, field3
"""

    response = llm.invoke(prompt).content

    # Parse the READY_TO_DRAFT flag
    ready = bool(re.search(r"READY_TO_DRAFT:\s*YES", response, re.IGNORECASE))

    # Parse missing fields if any
    missing = ""
    match = re.search(r"MISSING:\s*(.+)", response)
    if match:
        missing = match.group(1).strip()

    return {**state, "cot_analysis": response, "ready": ready, "missing": missing}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2: ask_questions
# Called when info is missing.
# Asks the user up to 3 focused questions.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ask_questions(state: State) -> State:
    prompt = f"""{SYSTEM}

The user wants a legal document but these details are missing:
{state['missing']}

Ask for the missing information politely and professionally.
Keep it to a maximum of 3 questions.

Format exactly like this:
"To draft your document accurately, I need a few details:
1. ...
2. ...
3. ..."
"""

    response = llm.invoke(prompt).content
    return {**state, "output": response}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3: draft_document
# Called when all info is present.
# Produces the full formatted legal document.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draft_document(state: State) -> State:
    prompt = f"""{SYSTEM}

Using the analysis below, draft the complete legal document.

Analysis:
{state['cot_analysis']}

Original Request:
{state['user_input']}

Use this exact structure:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[DOCUMENT TITLE IN CAPITALS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TO,
[Recipient Name & Address]

FROM,
[Sender Name & Address]

DATE: [DD/MM/YYYY]
PLACE: [City, State]

SUBJECT: [One-line subject]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PARTIES
2. BACKGROUND & FACTS
3. LEGAL PROVISIONS
4. GROUNDS / CLAUSES
5. RELIEF SOUGHT / PRAYER
6. NOTICE / DECLARATION
7. SIGNATURE BLOCK

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS (if any):

DISCLAIMER: This draft is for informational purposes only.
Consult a qualified advocate before filing or serving.
"""

    response = llm.invoke(prompt).content
    return {**state, "output": response}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTER
# Decides which node to go to after analyse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def router(state: State) -> Literal["draft_document", "ask_questions"]:
    return "draft_document" if state["ready"] else "ask_questions"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 — Build the graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

graph = StateGraph(State)

# Add nodes
graph.add_node("analyse",       analyse)
graph.add_node("ask_questions", ask_questions)
graph.add_node("draft_document", draft_document)

# Add edges
graph.add_edge(START, "analyse")
graph.add_conditional_edges("analyse", router)
graph.add_edge("ask_questions",  END)
graph.add_edge("draft_document", END)

# Compile
app = graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5 — Run it
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(user_input: str) -> str:
    """
    Pass any legal request as a string.
    Returns either the draft document or clarifying questions.
    """
    initial_state: State = {
        "user_input":   user_input,
        "cot_analysis": "",
        "ready":        False,
        "missing":      "",
        "output":       "",
    }
    result = app.invoke(initial_state)
    return result["output"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chat loop (run from terminal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Indian Legal Drafting Assistant")
    print("="*50)
    print("Type your request. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        print("\nAssistant:\n")
        print(run(user_input))
        print("\n" + "-"*50 + "\n")