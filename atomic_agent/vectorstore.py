import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in resp.data])

class LocalVectorStore:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.vectors = embed(docs)

    def search(self, query: str, k: int = 3) -> list[str]:
        q_vec = embed([query])[0]
        scores = self.vectors @ q_vec
        top = scores.argsort()[-k:][::-1]
        return [self.docs[i] for i in top]
