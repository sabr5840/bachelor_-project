import json
import math
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- Paths ---
SOURCE_DIR = Path("data/source_documents")
CHUNKS_DIR = Path("data/processed/chunks")
EMBEDDINGS_DIR = Path("data/processed/embeddings")

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_FILE = CHUNKS_DIR / "chunks.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "chunk_embeddings.json"

# --- Gemini client ---
client = genai.Client()

# Google documents the current embedding models in the Gemini API docs.
# We pick the stable text embedding model for a simple bachelor-project setup.
EMBED_MODEL = "gemini-embedding-001"


def read_source_documents() -> List[Dict[str, Any]]:
    """
    Reads all .txt files from source_documents recursively.
    Returns a list of document dictionaries with metadata.
    """
    documents = []

    for file_path in SOURCE_DIR.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            {
                "document_id": str(file_path.relative_to(SOURCE_DIR)).replace("/", "_").replace(".txt", ""),
                "title": file_path.stem.replace("_", " ").title(),
                "source_folder": str(file_path.parent.relative_to(SOURCE_DIR)),
                "filename": file_path.name,
                "text": text,
            }
        )

    return documents


def chunk_text(text: str, chunk_size_words: int = 120, overlap_words: int = 20) -> List[str]:
    """
    Splits text into overlapping chunks by word count.
    Overlap helps preserve context across chunk boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def create_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates chunk records from source documents.
    """
    all_chunks = []

    for doc in documents:
        chunk_texts = chunk_text(doc["text"], chunk_size_words=120, overlap_words=20)

        for i, text in enumerate(chunk_texts, start=1):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['document_id']}_chunk_{i}",
                    "document_id": doc["document_id"],
                    "document_title": doc["title"],
                    "source_folder": doc["source_folder"],
                    "filename": doc["filename"],
                    "text": text,
                }
            )

    return all_chunks


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks_file() -> None:
    """
    Reads source documents and only regenerates chunks for new or changed files.
    Existing chunks are reused if the source text has not changed.
    """
    documents = read_source_documents()

    existing_chunks = []
    if CHUNKS_FILE.exists():
        existing_chunks = load_json(CHUNKS_FILE)

    existing_by_document_id = {}
    for chunk in existing_chunks:
        document_id = chunk["document_id"]
        existing_by_document_id.setdefault(document_id, []).append(chunk)

    updated_chunks = []

    for doc in documents:
        document_id = doc["document_id"]
        existing_doc_chunks = existing_by_document_id.get(document_id, [])

        existing_full_text = (
            existing_doc_chunks[0].get("document_text")
            if existing_doc_chunks
            else None
        )

        if existing_full_text == doc["text"]:
            print(f"Reusing unchanged chunks: {document_id}")
            updated_chunks.extend(existing_doc_chunks)
            continue

        print(f"Creating chunks for new or changed document: {document_id}")

        chunk_texts = chunk_text(doc["text"], chunk_size_words=120, overlap_words=20)

        for i, text in enumerate(chunk_texts, start=1):
            updated_chunks.append(
                {
                    "chunk_id": f"{doc['document_id']}_chunk_{i}",
                    "document_id": doc["document_id"],
                    "document_title": doc["title"],
                    "source_folder": doc["source_folder"],
                    "filename": doc["filename"],
                    "text": text,
                    "document_text": doc["text"],
                }
            )

    save_json(CHUNKS_FILE, updated_chunks)

    print(f"Loaded {len(documents)} documents")
    print(f"Saved {len(updated_chunks)} chunks")


def embed_text(text: str, task_type: str) -> List[float]:
    """
    Generates an embedding vector for text.
    """
    # Google documents Gemini embedding models and embedding endpoints in the API docs.
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={
            "task_type": task_type
        },
    )
    return response.embeddings[0].values


def build_embeddings_file() -> None:
    """
    Loads chunks.json and reuses embeddings only if the chunk text is unchanged.
    New or changed chunks are embedded again.
    """
    chunks = load_json(CHUNKS_FILE)

    existing_embedded_chunks = []
    if EMBEDDINGS_FILE.exists():
        existing_embedded_chunks = load_json(EMBEDDINGS_FILE)

    existing_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in existing_embedded_chunks
    }

    updated_embedded_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk["chunk_id"]
        existing_chunk = existing_by_id.get(chunk_id)

        if existing_chunk and existing_chunk.get("text") == chunk["text"]:
            print(f"Skipping unchanged chunk {idx}/{len(chunks)}: {chunk_id}")
            updated_embedded_chunks.append(existing_chunk)
            continue

        print(f"Embedding new or changed chunk {idx}/{len(chunks)}: {chunk_id}")

        embedding = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")

        updated_embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding,
            }
        )

    save_json(EMBEDDINGS_FILE, updated_embedded_chunks)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Simple cosine similarity for semantic search.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_top_chunks(query: str, top_k: int = 3, min_score: float = 0.55) -> List[Dict[str, Any]]:
    embedded_chunks = load_json(EMBEDDINGS_FILE)
    query_embedding = embed_text(query, task_type="RETRIEVAL_QUERY")

    scored = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])

        if score >= min_score:
            scored.append(
                {
                    "score": score,
                    "chunk": chunk,
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = [item["chunk"] for item in scored[:top_k]]
    return top_chunks


def build_context(top_chunks: List[Dict[str, Any]]) -> str:
    """
    Creates context text for the Gemini prompt.
    """
    parts = []
    for chunk in top_chunks:
        parts.append(
            f"Kilde: {chunk['document_title']} ({chunk['filename']})\n"
            f"Indhold: {chunk['text']}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    # 1) Build chunk file
    build_chunks_file()

    # 2) Build embeddings file
    build_embeddings_file()

    # 3) Quick retrieval test
    test_query = "Hvad er forskellen på ratepension og livrente?"
    results = retrieve_top_chunks(test_query, top_k=3)

    print("\nTop chunks:")
    for chunk in results:
        print(f"- {chunk['chunk_id']} | {chunk['document_title']}")
        print(chunk["text"][:250])
        print()