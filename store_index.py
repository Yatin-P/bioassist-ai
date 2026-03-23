from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

print("Running store_index.py...")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

print("Index name:", INDEX_NAME)
print("API key loaded:", bool(PINECONE_API_KEY))

extracted_data = load_pdf_file("data/")
print("Documents loaded:", len(extracted_data))

text_chunks = text_split(extracted_data)
print("Text chunks created:", len(text_chunks))

embeddings = download_hugging_face_embeddings()
print("Embeddings model loaded")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("New Pinecone index created")

vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Data successfully stored in Pinecone")