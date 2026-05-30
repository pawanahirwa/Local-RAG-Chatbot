import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# ── Models (loaded once, CPU-only to save RAM) ──────────────────────────────
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 8},   # small batch = low memory spikes
)

# num_ctx kept small = less RAM + faster on gemma:2b
llm = OllamaLLM(model="gemma:2b", num_ctx=2048, temperature=0.2)

db = None


# ── Backend ─────────────────────────────────────────────────────────────────
def upload_pdf(pdf):
    global db

    if pdf is None:
        return "⚠️ No file selected.", gr.update(interactive=False)

    try:
        docs = PyPDFLoader(pdf.name).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(docs)

        db = Chroma.from_documents(chunks, embedding)

        return (
            f"✅ Ready! {len(docs)} pages → {len(chunks)} chunks indexed.",
            gr.update(interactive=True),
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


def rag_chat(query, history):
    global db

    if not query.strip():
        return history, ""

    if db is None:
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "⚠️ Please upload a PDF first."},
        ]
        return history, ""

    try:
        # Only 3 chunks -> small prompt -> fast inference + low RAM
        results = db.similarity_search(query, k=3)

        if not results:
            history = history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "No relevant content found in the document."},
            ]
            return history, ""

        context = "\n\n".join(d.page_content for d in results)

        # Short, simple prompt — gemma:2b handles these best
        prompt = f"""Use the context to answer the question. Be concise.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}
Answer:"""

        response = llm.invoke(prompt)
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.strip()},
        ]
        return history, ""

    except Exception as e:
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"❌ Error: {str(e)}"},
        ]
        return history, ""


def clear_chat():
    return [], ""


# ── Frontend ────────────────────────────────────────────────────────────────
css = """
#upload-col { border-right: 1px solid #e5e7eb; padding-right: 16px; }
#chatbot { min-height: 420px; }
footer { display: none !important; }
"""

with gr.Blocks(title="RAG Chatbot") as demo:

    gr.Markdown("## 📄 Local RAG Chatbot")
    gr.Markdown("Upload a PDF, then ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1, elem_id="upload-col"):
            gr.Markdown("### 📂 Document")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("⚡ Process PDF", variant="primary")
            status_box = gr.Textbox(
                label="Status", value="No document loaded.", interactive=False
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(elem_id="chatbot", show_label=False, height=420)
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="Ask a question about your document…",
                    show_label=False, scale=5, interactive=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=[status_box, query_input])
    send_btn.click(rag_chat, inputs=[query_input, chatbot], outputs=[chatbot, query_input])
    query_input.submit(rag_chat, inputs=[query_input, chatbot], outputs=[chatbot, query_input])
    clear_btn.click(clear_chat, outputs=[chatbot, query_input])

demo.launch(theme=gr.themes.Soft(), css=css)

