# 📄 RAG PDF Q&A

A local **Retrieval-Augmented Generation (RAG)** app that lets you upload PDFs and ask questions about their content.

## 🛠️ Tech Stack

* Python
* Streamlit
* Ollama — local LLM & embeddings
* ChromaDB — vector database
* PyMuPDF — PDF extraction
* LangChain — document processing
* Sentence Transformers — re-ranking

## 🔄 How It Works

```text
PDF
 ↓
Extract & Split
 ↓
Ollama Embeddings
 ↓
ChromaDB
 ↓
Semantic Search
 ↓
CrossEncoder Re-ranking
 ↓
Llama 3.2
 ↓
Answer
```

## 🚀 Setup

### 1. Create environment

```bash
python3.11 -m venv myvenv
source myvenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## ✨ Features

* Upload and process PDFs
* Semantic document search
* Cross-encoder re-ranking
* Local AI responses
* Persistent ChromaDB storage
* No cloud AI API required

## 🔮 Future Improvements

* Multiple document support
* Source/page citations
* Document management
* More file formats
* Model selection
* Chat history persistence

## 👨‍💻 Author

**creepahh**
