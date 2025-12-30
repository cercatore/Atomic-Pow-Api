from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from atomic_agent.loader import load_documents
from atomic_agent.vectorstore import LocalVectorStore
from atomic_agent.prompt import generate_prompt
from atomic_agent.agent import agent
from atomic_agents import BasicChatInputSchema
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field

# -------------------------------------------------
# Schemi input/output (model_config incluso)
# -------------------------------------------------

class ChatInputSchema(BaseIOSchema):
    """
    Input schema for chat messages from the user.
    """
    chat_message: str = Field(..., description="The user's input message")

    model_config = {
        "title": "ChatInputSchema",
        "description": "Input schema for chat messages from the user."
    }

class ChatOutputSchema(BaseIOSchema):
    """
    Output schema for chat messages from the agent.
    """
    chat_message: str = Field(..., description="The agent's response message")

    model_config = {
        "title": "ChatOutputSchema",
        "description": "Output schema for chat messages from the agent."
    }

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------

app = FastAPI(
    title="Atomic Agent API",
    version="1.0.0",
)

# -------------------------------------------------
# Startup: carica documenti + vector store
# -------------------------------------------------

DOC_PATH = "docs/manuale.pdf"

try:
    documents = load_documents(DOC_PATH)
    vector_store = LocalVectorStore(documents)
except Exception as e:
    raise RuntimeError(f"Failed to initialize vector store: {e}")

# -------------------------------------------------
# Request / Response models FastAPI
# -------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    suggested_questions: list[str]

# -------------------------------------------------
# Endpoint POST /query
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
