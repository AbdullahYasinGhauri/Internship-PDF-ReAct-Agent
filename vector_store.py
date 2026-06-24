import chromadb
import uuid

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="pdf_documents"
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def store_pdf(file_id: str, text: str):

    chunks = splitter.split_text(text)

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"file_id": file_id}
            for _ in chunks
        ]
    )


def search_pdf(file_id: str, query: str, k: int = 5):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"file_id": file_id}
    )

    docs = results["documents"][0]

    return "\n\n".join(docs)