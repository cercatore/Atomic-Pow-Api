def generate_prompt(context_docs: list[str], query: str) -> str:
    context = "\n\n".join(context_docs)
    return f"""
Usa SOLO il contesto seguente.

CONTESTO:
{context}

DOMANDA:
{query}
"""
