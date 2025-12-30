from dotenv import load_dotenv
load_dotenv()  # carica .env nel process env
from atomic_agent.loader import load_documents
from atomic_agent.vectorstore import LocalVectorStore
from atomic_agent.prompt import generate_prompt
from atomic_agent.agent import agent
from atomic_agents import BasicChatInputSchema

def main():
    docs = load_documents("docs/manuale.pdf")
    store = LocalVectorStore(docs)

    query = "Come funziona atomic agents?"
    retrieved_docs = store.search(query, k=3)

    prompt = generate_prompt(retrieved_docs, query)

    response = agent.run(
        BasicChatInputSchema(chat_message=prompt)
    )

    print("\n=== RISPOSTA ===")
    print(response.chat_message)

    print("\n=== FOLLOW-UP ===")
    for q in response.suggested_questions:
        print("-", q)

if __name__ == "__main__":
    main()
