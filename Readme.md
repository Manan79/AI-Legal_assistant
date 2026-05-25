# AI Legal Assistant

A FastAPI backend for an AI-powered Indian legal assistant built with LangGraph, LangChain, AstraDB, and Google Generative AI.

## Overview

This project provides an API wrapper around a LangGraph workflow for legal reasoning, retrieval-augmented generation, and document drafted.

Key capabilities:
- Process legal queries through a LangGraph workflow
- Retrieve legal context from an AstraDB vector store
- Search Wikipedia and web sources for additional context
- Draft legal documents using a generative drafting tool
- Expose `/query`, `/chat`, and `/health` endpoints via FastAPI

## Architecture

- `main.py` - FastAPI app exposing the legal assistant endpoints
- `graph.py` - Compiles the workflow graph and constructs the legal agent pipeline
- `base.py` - Defines the workflow input schema and LangChain tracing configuration
- `functionalities.py` - Defines tools, system prompt, and agent configuration
- `db_connection.py` - Connects to AstraDB vector store and initializes embeddings/model
- `drafter.py` - Drafts legal documents using Google Generative AI

## Requirements

Install Python dependencies from:

```bash
pip install -r requirements.txt
```

The project relies on:
- FastAPI / Uvicorn
- LangChain and LangGraph
- AstraDB vector search
- Hugging Face sentence-transformers embeddings
- Google Generative AI for drafting
- Wikipedia and web search utilities

## Setup

1. Clone the repository.
2. Create a `.env` file in the project root with the required environment variables.

Suggested variables:

```env
ASTRA_DB_API_ENDPOINT=<your_astra_db_api_endpoint>
ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_application_token>
HF_HOME=./hf_cache
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

- Health check: `GET /health`
- Legal query endpoint: `POST /query`
- Chat endpoint: `POST /chat`

Example request body:

```json
{
  "query": "Generate a timeline for the Arvind Kejriwal CBI case with a short summary."
}
```

Example response:

```json
{
  "status": "success",
  "query": "...",
  "response": "...",
  "timestamp": "2026-05-25T00:00:00"
}
```

## Notes

- The project uses local Hugging Face caching under `hf_cache/`.
- Draft templates and generated documents can be stored under `drafted_agreements/`.
- The workflow is currently a single-node graph with a legal reasoning agent, but it can be extended by adding more nodes and tools in `graph.py`.

## Development

- Update the system prompt in `functionalities.py` to refine legal behavior.
- Add additional tools or retrieval sources as needed.
- Use `react_rag.ipynb` for experimentation and prototyping.

## License

This repository does not include a license file. Add a license if you plan to share or distribute the code.
