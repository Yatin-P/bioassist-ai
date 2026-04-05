from flask import Flask, render_template, request, session, Response, stream_with_context
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "bioassist_secret_key"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})


@app.route("/")
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "current_topic" not in session:
        session["current_topic"] = ""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def stream_chat():
    user_message = request.form["msg"]

    # Fresh OpenAI client per request
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Session memory
    chat_history = session.get("chat_history", [])
    current_topic = session.get("current_topic", "")

    recent_history = chat_history[-4:]
    history_text = "\n".join(
        [f"User: {item['user']}\nBot: {item['bot']}" for item in recent_history]
    )

    # Smarter retrieval query
    if current_topic and history_text.strip():
        retrieval_query = f"""
Current topic: {current_topic}

Current user question:
{user_message}

Recent conversation:
{history_text}
"""
    elif history_text.strip():
        retrieval_query = f"""
Current user question:
{user_message}

Recent conversation:
{history_text}
"""
    else:
        retrieval_query = user_message

    docs = retriever.invoke(retrieval_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        filename = source.split("/")[-1]
        if filename not in sources:
            sources.append(filename)

    topic_prompt = f"""
Detect the main medical topic.

Recent conversation:
{history_text}

Question:
{user_message}

Return only topic name.
"""

    topic_response = client.responses.create(
        model="gpt-5-nano",
        input=topic_prompt
    )

    detected_topic = topic_response.output_text.strip()
    if detected_topic:
        current_topic = detected_topic
        session["current_topic"] = current_topic

    prompt = f"""
You are BioAssist AI, a smart and helpful medical study assistant.

Your job:
- Answer only from the provided context and recent conversation.
- Use the current topic to understand follow-up questions.
- Do not guess.
- Do not make up facts.
- If partial information exists in the context, answer using that information.
- Only say "I don't know based on the provided documents." if nothing relevant exists.

Response rules:
- Be clear, accurate, and student-friendly.
- Write in a polished chatbot style, like a helpful tutor.
- Keep the answer concise but informative.
- Use short paragraphs and bullet points where useful.
- Do not mention anything outside the provided context.

Formatting rules:
- Start with a direct answer.
- Then use helpful sections when relevant.

For definition questions:
Definition:
Explanation:
Key Points:

For comparison questions:
Overview:
Main Differences:
Key Points:

For process or mechanism questions:
Overview:
How It Works:
Key Points:

For importance/function questions:
Answer:
Why It Matters:
Key Points:

For broad medical topic questions:
Overview:
Important Details:
Key Points:

Current Topic:
{current_topic}

Recent Conversation:
{history_text}

Context:
{context}

Question:
{user_message}
"""

    def generate():
        final_answer = ""

        with client.responses.stream(
            model="gpt-5-nano",
            input=prompt
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    chunk = event.delta
                    final_answer += chunk
                    yield chunk

        if sources:
            source_text = "\n\n---\nSources used: " + ", ".join(sources)
            final_answer += source_text
            yield source_text

        chat_history.append({
            "user": user_message,
            "bot": final_answer
        })
        session["chat_history"] = chat_history[-10:]

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/clear", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    session["current_topic"] = ""
    return "Chat cleared"


if __name__ == "__main__":
    app.run(debug=True)