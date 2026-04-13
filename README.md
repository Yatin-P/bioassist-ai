# BioAssist AI

BioAssist AI is a medical document question-answering chatbot built with Flask, Pinecone, embeddings, and OpenAI.

## Features
- Login/logout authentication for app access
- Ask questions from a medical PDF
- Store document embeddings in Pinecone
- Retrieve relevant document chunks
- Generate grounded answers with OpenAI
- Simple Flask chat interface

## Tech Stack
- Python
- Flask
- Pinecone
- OpenAI
- LangChain

## Project Structure
- `app.py` - main Flask application
- `store_index.py` - indexes PDF data into Pinecone
- `src/helper.py` - PDF loading, chunking, embeddings
- `src/prompt.py` - prompt template
- `templates/login.html` - login page
- `templates/chat.html` - frontend chat UI

## Setup
1. Create and activate a virtual environment
2. Install dependencies
3. Add environment variables in `.env`
4. Run `python store_index.py`
5. Run `python app.py`

## Environment Variables
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `OPENAI_API_KEY`
- `BIOASSIST_USERNAME` (optional, default: `admin`)
- `BIOASSIST_PASSWORD` (optional, default: `bioassist123`)

## Run
```bash
python app.py
```

Then open the app and log in with your configured credentials.
