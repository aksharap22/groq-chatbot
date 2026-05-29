# Groq RAG Chatbot using Streamlit

A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit and Groq API.

The application can:
- Chat normally like a standard AI assistant
- Answer questions from uploaded PDF and TXT documents
- Perform semantic retrieval using embeddings
- Handle large documents efficiently using RAG

---

## Features

- Conversational AI chatbot
- Normal chat mode
- PDF and TXT document support
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- Source chunk retrieval
- Groq LLM integration
- Adjustable chunk size and retrieval settings
- Chat history support
- Simple Streamlit interface
- Bring Your Own API Key support

---

## How It Works

```text
User Query
    ↓
Retrieve Relevant Document Chunks
    ↓
Send Context to Groq LLM
    ↓
Generate Grounded Response
```

If no document is uploaded, the chatbot works as a normal conversational AI assistant.

---

## Tech Stack

- Python
- Streamlit
- Groq API
- Sentence Transformers
- NumPy
- PDFPlumber

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/groq-rag-chatbot.git
cd groq-rag-chatbot
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## Get Groq API Key

1. Visit https://console.groq.com
2. Create an account
3. Generate an API key
4. Paste the key into the sidebar

---

## Run the App

```bash
streamlit run app.py
```

Open browser at:

```text
http://localhost:8501
```

---

## Supported File Types

- PDF (.pdf)
- Text files (.txt)

---

## Project Structure

```text
groq-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Models Supported

- llama-3.3-70b-versatile
- llama3-8b-8192
- mixtral-8x7b-32768

---

## Core Components

| Component | Purpose |
|---|---|
| PDFPlumber | Extract text from PDFs |
| Sentence Transformers | Generate embeddings |
| NumPy | Vector operations |
| Groq API | LLM inference |
| Streamlit | Frontend UI |

---

## Example Use Cases

- General AI chatbot
- Research assistant
- PDF question answering
- Notes summarization
- Resume analysis
- Knowledge base assistant
- Academic document chat

---

## Future Improvements

- FAISS or ChromaDB integration
- Multi-document support
- Persistent vector storage
- Streaming responses
- Citation highlighting
- Chat export
- Cloud deployment

---

## Deployment

Deploy easily using:
- Streamlit Community Cloud
- GitHub

---

## Author

Akshara Perala