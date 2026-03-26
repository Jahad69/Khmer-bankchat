# 🏦 Khmer BankChat — Dual-Model Inference Interface

A production-ready chat interface for Khmer banking Q&A with conversational memory and Retrieval-Augmented Generation (RAG).

This project features a FastAPI backend and a Streamlit frontend, allowing live switching between fine-tuned Qwen and SeaLLMs adapters at runtime without restarting.

---

## 🏗️ Technical Architecture

### Models & Adapters
The application uses **QLoRA (4-bit quantization)** to run large language models efficiently within limited VRAM scenarios:

| Adapter ID | Base Model | Adapter Size | LoRA Rank | Purpose |
|------------|------------|--------------|-----------|---------|
| `qwen` | `unsloth/qwen3-4b-unsloth-bnb-4bit` | ~126 MB | r=16 | Fast, lightweight reasoning |
| `seallm` | `SeaLLMs/SeaLLMs-v3-7B-Chat` | ~308 MB | r=64 | High-quality Khmer NLP |

**Key Design Features:**
1. **Dynamic Loading:** The models are loaded via `peft.PeftModel.from_pretrained()` wrapped around a 4-bit `bitsandbytes` quantized base model, placed directly on the GPU.
2. **Thread-Safe Generation:** The `model_manager.py` uses a thread-lock singleton to ensure concurrent requests avoid memory corruption during generation or adapter switching.
3. **Streaming Support:** Tokens are streamed to the frontend via Server-Sent Events (SSE).

### Conversational Memory & History Logic
The interface maintains context across multi-turn conversations through a coordinated effort between the frontend and backend:

1. **Frontend State Storage:** The Streamlit app (`app.py`) stores the entire conversation history in `st.session_state.messages`.
2. **Context Payload Delivery:** On each new prompt, the UI bundles the `system_prompt` and the **last 20 messages** into the JSON payload sent to either `/chat` or `/chat/stream`.
3. **Backend Parsing Protocol (ChatML):** The FastAPI backend (`model_manager._build_prompt`) handles tokenization using the adapter's ChatML template (`<|im_start|>` and `<|im_end|>`). If the localized model lacks its Jinja template, a robust fallback automatically formats the chronological message history into strict ChatML format. This guarantees that LoRA adapters receive properly structured contextual memory aligned with their fine-tuning dataset.
4. **RAG Injection:** RAG context hits (from the FAISS index) are *only* injected into the **last** user message dynamically to supply targeted knowledge without distorting the historical flow. 

---

## 🚀 Setup Instructions

### Environment & Dependencies
Require **Python 3.10+** and a CUDA-capable GPU for BitsAndBytes quantization.

```bash
# 1. Navigate to the project directory
cd /home/kdor/CADT/Intern2/Code/prepare-inference

# 2. Create and activate a Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

*(Optional: If running on a CPU-only machine, remove `bitsandbytes` from requirements and load models with "none" quantization in the UI).*

### Starting the Application

We provide helper scripts to launch the services. It is recommended to run them in the background using `start_all.sh`:

```bash
# Make scripts executable
chmod +x start_all.sh stop_all.sh start_api.sh start_ui.sh

# Start backend (API) and frontend (Streamlit)
./start_all.sh
```

Then, open **http://localhost:8501** in your web browser.

Alternatively, to start the services manually in separate terminals:

**Terminal A (Backend API server):**
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal B (Frontend UI):**
```bash
streamlit run frontend/app.py
```

---

## 📁 Repository Structure

```
prepare-inference/
├── backend/
│   ├── api.py            ← FastAPI framework (HTTP endpoints & SSE streaming)
│   ├── model_manager.py  ← Adapter lifecycle, ChatML prompting, and token generation
│   └── rag_engine.py     ← FAISS indexing, Khmer segmentation, and retrieval
├── frontend/
│   └── app.py            ← Interactive Streamlit chat UI
├── qwen-model-ok/        ← Fine-tuned Qwen3-4B adapter weights
├── SeaLLM-model-ok/      ← Fine-tuned SeaLLMs-v3-7B adapter weights
├── requirements.txt      ← Environment dependencies
└── *.sh                  ← Startup and shutdown bash shell helper scripts
```

---

## ⚙️ RAG Engine Configuration

1. In the sidebar, paste the **absolute path** to your Knowledge Base CSV (must contain `Question` and `Answer` headers).
2. Click **Load KB** to generate the FAISS index embeddings (using `.paraphrase-multilingual-mpnet-base-v2` & `khmernltk`).
3. Turn on the **Use RAG** toggle switch.

When enabled, the engine retrieves the top-K semantically aligned Q&A pairs and integrates them temporarily as implicit instructions on the latest prompt layer.

## 🔌 API Reference (Port 8000)

- **GET `/adapters`**: Discover loaded weights
- **GET `/status`**: Identify active running models & hardware capability
- **POST `/load_model`**: Switch Base/LoRA instances efficiently 
- **POST `/chat`**: Sync text-generation
- **POST `/chat/stream`**: SSE text-generation stream

Interactive Swagger documentation available at **http://localhost:8000/docs**.
