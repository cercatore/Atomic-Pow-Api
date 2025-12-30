from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from atomic_agent.loader import load_documents
from atomic_agent.vectorstore import LocalVectorStore
from atomic_agent.prompt import generate_prompt
from atomic_agent.agent import agent

from atomic_agents import BasicChatInputSchema

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="Atomic Agent API",
    version="1.0.0",
)

# -------------------------------------------------
# Startup: load docs + build vector store ONCE
# -------------------------------------------------

DOC_PATH = "docs/manuale.pdf"

try:
    documents = load_documents(DOC_PATH)
    vector_store = LocalVectorStore(documents)
except Exception as e:
    raise RuntimeError(f"Failed to initialize vector store: {e}")

# -------------------------------------------------
# Request / Response models
# -------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    suggested_questions: list[str]

# -------------------------------------------------
# API endpoint
# -------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_agent(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    retrieved_docs = vector_store.search(
        payload.query,
        k=payload.top_k
    )

    prompt = generate_prompt(retrieved_docs, payload.query)

    response = agent.run(
        BasicChatInputSchema(chat_message=prompt)
    )

    return QueryResponse(
        answer=response.chat_message,
        suggested_questions=response.suggested_questions
    )
