import sys
from dotenv import load_dotenv
load_dotenv()  # carica .env nel process env

from atomic_agent.loader import load_documents
from atomic_agent.vectorstore import LocalVectorStore
from atomic_agent.prompt import generate_prompt
from atomic_agent.agent import agent
from atomic_agents import BasicChatInputSchema
from pydantic import Field
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

# -------------------------------------------------
# Schemi con docstring + model_config per atomic-agents
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
# Funzione main
# -------------------------------------------------

def main():
    # Documento passato come argomento shell, default fallback
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "docs/manuale.pdf"

    # Carica documenti
    docs = load_documents(doc_path)
    store = LocalVectorStore(docs)

    # Query di esempio
    query = "Come funziona atomic agents?"
    retrieved_docs = store.search(query, k=3)

    # Genera prompt
    prompt = generate_prompt(retrieved_docs, query)

    # Esegui l'agent
    response = agent.run(
        BasicChatInputSchema(chat_message=prompt)
    )

    # Stampa risultati
    print("\n=== RISPOSTA ===")
    print(response.chat_message)

    print("\n=== FOLLOW-UP ===")
    for q in response.suggested_questions:
        print("-", q)

# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    main()
