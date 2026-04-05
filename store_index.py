from src.helper import load_pdf_file, text_split
from dotenv import load_dotenv
import os
import uuid

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

print("Running store_index.py...")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Index name:", INDEX_NAME)
print("API key loaded:", bool(PINECONE_API_KEY))

extracted_data = load_pdf_file("data/")
print("Documents loaded:", len(extracted_data))

text_chunks = text_split(extracted_data)
print("Text chunks created:", len(text_chunks))

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
print("OpenAI embeddings loaded")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("New Pinecone index created")

index = pc.Index(INDEX_NAME)

vectors = []
batch_size = 50

for i in range(0, len(text_chunks), batch_size):
    chunk_batch = text_chunks[i:i + batch_size]
    texts = [doc.page_content[:8000] for doc in chunk_batch]  # safety trim
    embedded_batch = embeddings.embed_documents(texts)

    for doc, embedding in zip(chunk_batch, embedded_batch):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": doc.page_content[:8000],
                "source": doc.metadata.get("source", "unknown")
            }
        })

    print(f"Embedded batch {i // batch_size + 1}")

upsert_batch_size = 100
for i in range(0, len(vectors), upsert_batch_size):
    batch = vectors[i:i + upsert_batch_size]
    index.upsert(vectors=batch)
    print(f"Uploaded batch {i // upsert_batch_size + 1}")

print("Data successfully stored in Pinecone")