from pathlib import Path
from pypdf import PdfReader

def load_documents(path: str) -> list[str]:
    p = Path(path)
    docs = []

    if p.suffix == ".pdf":
        reader = PdfReader(p)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                docs.append(text)

    elif p.suffix in [".txt", ".md"]:
        docs.append(p.read_text())

    else:
        raise ValueError("Formato non supportato")

    return docs
