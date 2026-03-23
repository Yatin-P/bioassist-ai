from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Embeddings model (same one used during indexing)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing vector store
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    # Retrieve relevant chunks from Pinecone
    docs = retriever.invoke(user_message)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful medical study assistant.
Answer the user's question only from the provided context.
If the answer is not in the context, say: "I don't know based on the provided document."

Context:
{context}

Question:
{user_message}
"""

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":
    app.run(debug=True)