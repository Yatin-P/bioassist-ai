from functools import wraps
import os
import sqlite3
import time

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from pypdf import PdfReader

app = Flask(__name__)
app.secret_key = "bioassist_secret_key"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_DB_PATH = os.getenv("BIOASSIST_USER_DB", "users.db")
UPLOAD_FOLDER = os.getenv("BIOASSIST_UPLOAD_DIR", "uploaded_docs")
MAX_UPLOAD_CHARS = 20000
MAX_ATTACHED_FILES = 5

USERS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def initialize_auth_storage():
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.execute(USERS_TABLE_DDL)
        conn.commit()


def initialize_upload_storage():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_document_text(file_storage):
    filename = file_storage.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension in {"txt", "md"}:
        text = file_storage.read().decode("utf-8", errors="ignore")
    elif extension == "pdf":
        reader = PdfReader(file_storage)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return None

    cleaned = " ".join(text.split())
    return cleaned[:MAX_UPLOAD_CHARS].strip()


def fetch_user(username):
    initialize_auth_storage()
    normalized_username = username.strip()
    with sqlite3.connect(USER_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
    return user


def register_user(username, password):
    initialize_auth_storage()
    normalized_username = username.strip()
    password_hash = generate_password_hash(password)
    try:
        with sqlite3.connect(USER_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (normalized_username, password_hash),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("logged_in"):
        return redirect(url_for("index_page"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return render_template("register.html")
        if password != confirm_password:
            flash("Passwords do not match.")
            return render_template("register.html")

        if not register_user(username, password):
            flash("That username is already taken.")
            return render_template("register.html")

        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index_page"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = fetch_user(username)

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["logged_in"] = True
            session["username"] = user["username"]
            session["chat_history"] = []
            session["current_topic"] = ""
            return redirect(url_for("index_page"))

        flash("Invalid username or password.")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/upload-document", methods=["POST"])
@login_required
def upload_document():
    if "document" not in request.files:
        return jsonify({"message": "No file selected."}), 400

    file = request.files["document"]
    if not file or not file.filename:
        return jsonify({"message": "No file selected."}), 400

    text = extract_document_text(file)
    if not text:
        return jsonify({"message": "Unsupported or empty file. Use PDF, TXT, or MD."}), 400

    username = session.get("username", "user")
    safe_name = secure_filename(file.filename)
    saved_filename = f"{username}_{int(time.time())}_{safe_name}.txt"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)

    with open(saved_path, "w", encoding="utf-8") as f:
        f.write(text)

    attached_documents = session.get("attached_documents", [])
    attached_documents.append({"name": safe_name, "path": saved_path})
    attached_documents = attached_documents[-MAX_ATTACHED_FILES:]
    session["attached_documents"] = attached_documents

    return jsonify(
        {
            "message": f"Attached: {safe_name}",
            "documents": attached_documents,
        }
    )


@app.route("/remove-document", methods=["POST"])
@login_required
def remove_document():
    name = request.form.get("name", "")
    attached_documents = session.get("attached_documents", [])

    updated_documents = [doc for doc in attached_documents if doc.get("name") != name]
    session["attached_documents"] = updated_documents

    return jsonify({"documents": updated_documents})


@app.route("/")
@login_required
def index_page():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "current_topic" not in session:
        session["current_topic"] = ""
    return render_template("chat.html", username=session.get("username", "User"))


@app.route("/get", methods=["POST"])
@login_required
def stream_chat():
    user_message = request.form["msg"]

    chat_history = session.get("chat_history", [])
    current_topic = session.get("current_topic", "")
    attached_documents = session.get("attached_documents", [])
    attached_doc_context_blocks = []

    for doc in attached_documents:
        doc_name = doc.get("name", "")
        doc_path = doc.get("path", "")
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read(MAX_UPLOAD_CHARS)
            attached_doc_context_blocks.append(f"[{doc_name}]\n{text}")

    attached_doc_context = "\n\n".join(attached_doc_context_blocks)

    recent_history = chat_history[-4:]
    history_text = "\n".join(
        [f"User: {item['user']}\nBot: {item['bot']}" for item in recent_history]
    )

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

    query_embedding = embeddings.embed_query(retrieval_query)

    results = index.query(vector=query_embedding, top_k=6, include_metadata=True)

    matches = results.get("matches", [])

    context_parts = []
    sources = []

    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        source = metadata.get("source", "Unknown source")
        filename = source.split("/")[-1]

        if text:
            context_parts.append(text)

        if filename not in sources:
            sources.append(filename)

    context = "\n\n".join(context_parts)

    topic_prompt = f"""
Detect the main medical topic.

Recent conversation:
{history_text}

Question:
{user_message}

Return only topic name.
"""

    topic_response = client.responses.create(model="gpt-5-nano", input=topic_prompt)

    detected_topic = topic_response.output_text.strip()
    if detected_topic:
        current_topic = detected_topic
        session["current_topic"] = current_topic

    prompt = f"""
You are BioAssist AI, a smart and helpful medical study assistant.

Your job:
- Prioritize attached documents first when answering.
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

Attached Documents:
{attached_doc_context}

Question:
{user_message}
"""

    def generate():
        final_answer = ""

        with client.responses.stream(model="gpt-5-nano", input=prompt) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    chunk = event.delta
                    final_answer += chunk
                    yield chunk

        if sources:
            source_text = "\n\n---\nSources used: " + ", ".join(sources)
            final_answer += source_text
            yield source_text

        chat_history.append({"user": user_message, "bot": final_answer})
        session["chat_history"] = chat_history[-10:]

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/clear", methods=["POST"])
@login_required
def clear_chat():
    session["chat_history"] = []
    session["current_topic"] = ""
    return "Chat cleared"


initialize_auth_storage()
initialize_upload_storage()


if __name__ == "__main__":
    app.run(debug=True)
