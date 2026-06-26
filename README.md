# AI Legal Assistant

AI Legal Assistant is a FastAPI-based backend that uses LangGraph, LangChain, OpenAI, Google Gemini, Astra DB vector search, and optional web research tools to answer Indian legal queries and draft common legal documents.

The project is designed around a legal-focused agent that can:

- answer Indian law and case-related questions,
- retrieve supporting context from a vector database of legal material,
- fall back to Wikipedia or web search when needed,
- generate agreement or document drafts for common legal use cases.

## What This Project Does

This codebase exposes a backend API for a legal assistant workflow. A user submits a legal query, and the system routes that query into a LangGraph-powered agent with access to multiple tools:

- `retriever_store` for legal retrieval from Astra DB,
- `wikipedia_search` for general supporting context,
- `websearch` for more recent public information,
- `generate_draft` for legal document drafting.

The current implementation is focused on Indian legal reasoning, legal research, case-law assistance, and simple legal drafting.

## Key Features

- Legal-domain assistant restricted to law-related queries
- LangGraph workflow orchestration
- Vector retrieval using Astra DB and Hugging Face embeddings
- OpenAI chat model for agent reasoning
- Gemini-based drafting helper for agreements and legal documents
- FastAPI API with interactive Swagger docs
- Docker support for backend deployment
- Optional LangSmith tracing configuration

## Tech Stack

- Python 3.13
- FastAPI
- LangChain
- LangGraph
- OpenAI (`gpt-4o-mini`)
- Google Generative AI (`gemini-2.5-flash`)
- Astra DB Vector Store
- Hugging Face sentence embeddings (`all-MiniLM-L6-v2`)
- Tavily Search
- Wikipedia tool integration

## Project Structure

```text
AI-Legal-Assistant/
|-- backend/
|   |-- main.py                 # FastAPI app and API routes
|   |-- graph.py                # LangGraph workflow definition
|   |-- base.py                 # Shared agent state schema
|   |-- db_connection.py        # Model, embedding, and Astra DB setup
|   |-- functionalities.py      # Tools and main legal agent
|   |-- drafter.py              # Draft generation logic
|   |-- drafted_agreements/     # Saved template/example drafts
|   |-- evaluation.ipynb        # Notebook for experiments/evaluation
|   `-- hf_cache/               # Local embedding/model cache
|-- vectorstore/
|   |-- astradb.ipynb           # Notebook for vector DB work
|   `-- pdf_to_text.py          # Placeholder utility script
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Architecture Overview

### 1. API layer

`backend/main.py` defines the FastAPI server and exposes:

- `GET /` for a basic service description
- `GET /health` for health checks
- `POST /chat` for legal assistant queries

### 2. Workflow layer

`backend/graph.py` builds a very simple LangGraph workflow:

- start
- run one legal agent node
- end

The compiled workflow is stored in `new_workflow` and called by the API layer.

### 3. Agent layer

`backend/functionalities.py` creates the main legal agent using `create_agent(...)`.

The system prompt strongly constrains the assistant to:

- Indian legal topics,
- judgments and court proceedings,
- drafting and compliance tasks,
- factual caution around allegations and pending matters,
- grounded responses over hallucinated citations.

### 4. Retrieval layer

`backend/db_connection.py` connects the application to Astra DB and sets up:

- Hugging Face embeddings,
- the `AI_Legal_database` collection,
- a retriever using MMR search (`k=10`, `fetch_k=30`, `lambda_mult=0.7`).

### 5. Drafting layer

`backend/drafter.py` provides a drafting tool for documents such as:

- legal notices,
- affidavits,
- RTI applications,
- NDAs,
- rental agreements,
- consumer complaints,
- MOUs.

If a matching text file already exists in `backend/drafted_agreements/`, it returns that file. Otherwise it generates a new draft with Gemini and returns HTML converted from Markdown.

## Environment Variables

Create a `.env` file in the project root with the values your setup requires.

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_ai_key
TAVILY_API_KEY=your_tavily_key
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
LANGCHAIN_API_KEY=your_langsmith_key_optional
```

### Notes

- `OPENAI_API_KEY` is needed because the main agent uses `ChatOpenAI(model="gpt-4o-mini")`.
- `GOOGLE_API_KEY` is needed because the drafting tool uses `GoogleGenerativeAI(model="gemini-2.5-flash")`.
- `TAVILY_API_KEY` is needed if the web search tool is used.
- `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN` are required for vector retrieval.
- LangSmith tracing is partially configured in code; adding `LANGCHAIN_API_KEY` is recommended if you want tracing to work.

## Installation

### Option 1: Local setup with `pip`

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

If you want to follow the declared project metadata in `pyproject.toml`, make sure you are using Python 3.13 or newer.

### Option 2: Local setup with `uv`

```bash
uv sync
```

## Running the Backend

Start the API locally with:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Once the server is running:

- API root: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

## Docker

Build the container:

```bash
docker build -t ai-legal-assistant .
```

Run it:

```bash
docker run --env-file .env -p 8000:8000 ai-legal-assistant
```

The Docker image installs CPU-only PyTorch first, then installs the Python dependencies, and finally starts the FastAPI app with Uvicorn.

## API Usage

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "message": "AI Legal Assistant API is running",
  "timestamp": "2026-06-24T12:34:56.000000"
}
```

### Chat endpoint

```http
POST /chat
Content-Type: application/json
```

Request body:

```json
{
  "query": "Draft a rental agreement for a residential apartment in Delhi."
}
```

Example `curl` request:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Can you explain the legal difference between an allegation, charge, and conviction in Indian criminal law?\"}"
```

Example response shape:

```json
{
  "status": "success",
  "query": "Can you explain the legal difference between an allegation, charge, and conviction in Indian criminal law?",
  "response": "...",
  "timestamp": "2026-06-24T12:34:56.000000"
}
```

## How the Query Flow Works

1. A user sends a request to `POST /chat`.
2. The API wraps the user input as a LangChain `HumanMessage`.
3. The LangGraph workflow invokes the legal agent.
4. The agent decides whether to:
   - answer directly,
   - retrieve legal context from Astra DB,
   - search Wikipedia,
   - use web search,
   - generate a draft.
5. The final assistant message is returned as the API response.

## Drafting Behavior

The drafting tool supports common legal document generation. The current code also checks `backend/drafted_agreements/` first, so you can keep reusable draft templates there.

An example file already exists:

- `backend/drafted_agreements/Rental.txt`

This means a rental draft request may return the stored template instead of generating a new one.

## Data and Vector Store Notes

The project expects a pre-existing Astra DB collection named `AI_Legal_database`.

The retriever is intended to ground answers in legal content such as:

- Supreme Court judgments,
- legal clauses,
- legal references,
- supporting legal context.

The `vectorstore/` folder currently contains notebook-based work rather than a fully documented ingestion pipeline, so dataset preparation and indexing may still be partly manual.

## Development Notes

- `.env` is gitignored.
- `backend/hf_cache/` stores local Hugging Face model cache files.
- `backend/models.py` is currently empty.
- `vectorstore/pdf_to_text.py` is currently empty.
- The repository includes notebooks for experimentation and evaluation.

## Current Limitations

- The root endpoint mentions `POST /query`, but the implemented public route is `POST /chat`.
- The workflow is currently a single-agent-node graph, so orchestration is simple rather than multi-step.
- There are no automated tests in the repository yet.
- Some files appear to be placeholders or early scaffolding.
- The system depends on multiple external services, so local startup will fail without valid credentials.

## Suggested Improvements

- Add automated tests for API routes and tool behavior
- Add a documented ingestion pipeline for legal PDFs into Astra DB
- Add structured error handling for missing environment variables
- Separate configuration into a dedicated settings module
- Add response source attribution for retrieved legal material
- Add frontend or chat UI if this is meant to be user-facing

## Screenshots

<img width="1905" height="463" alt="Screenshot 2026-05-25 004103" src="https://github.com/user-attachments/assets/08cb33fd-30e2-4545-8886-7346e8378510" />
<img width="1075" height="357" alt="Screenshot 2026-05-25 004232" src="https://github.com/user-attachments/assets/0123be41-cd4e-46eb-8619-b17b8c2e21f2" />
<img width="1575" height="325" alt="Screenshot 2026-05-25 004331" src="https://github.com/user-attachments/assets/9dfe9403-796b-4c5e-ad84-c68e43023087" />
<img width="628" height="441" alt="Screenshot 2026-05-25 004628" src="https://github.com/user-attachments/assets/9c071208-6a37-4c6f-9deb-865a45de2de1" />


## License

No license file is currently present in the repository. Add one if you plan to distribute or open-source the project.
