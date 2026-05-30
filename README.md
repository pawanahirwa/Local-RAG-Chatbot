# 📄 Local RAG Chatbot

A lightweight, fully local **Retrieval-Augmented Generation (RAG)** chatbot. Upload a PDF and ask questions about it — answers are grounded in the document. Runs entirely on your machine using [Ollama](https://ollama.com), so no API keys or internet are needed for inference.

## ✨ Features

- 📂 Upload any PDF and chat with it
- 🔍 Semantic search over document chunks (ChromaDB + sentence-transformers)
- 🤖 Runs a local LLM via Ollama (`gemma:2b` by default)
- 💬 Clean two-panel Gradio chat interface
- 🪶 Optimized to run on low-RAM machines

## 🛠️ Tech Stack

- **Gradio** – web UI
- **LangChain** – RAG pipeline
- **ChromaDB** – vector store
- **sentence-transformers** (`all-MiniLM-L6-v2`) – embeddings
- **Ollama** – local LLM inference

## 📦 Setup

### 1. Install Ollama and pull a model

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull gemma:2b
```

> On very low RAM (under ~3 GB), use a smaller model instead:
> ```bash
> ollama pull qwen2.5:0.5b
> ```
> and change the model name in `app.py`.

### 2. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/local-rag-chatbot.git
cd local-rag-chatbot

python -m venv rag_env
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate

pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

Then open the local URL shown in your terminal (usually `http://127.0.0.1:7860`).

## 🚀 Usage

1. Upload a PDF in the left panel
2. Click **Process PDF** and wait for the "Ready!" status
3. Ask questions in the chat box on the right

## ⚙️ Configuration

Key settings in `app.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gemma:2b` | Ollama model to use |
| `chunk_size` | `700` | Characters per document chunk |
| `k` | `3` | Number of chunks retrieved per query |
| `num_ctx` | `2048` | LLM context window size |

## 📄 License

MIT
