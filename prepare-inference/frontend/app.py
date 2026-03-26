"""
app.py – Streamlit BankChat UI
===============================
A premium chat interface for the Khmer Banking Chatbot.
Talks to the FastAPI backend (api.py) via HTTP.

Run:
  # Tab 1 – start the API
  cd /home/kdor/CADT/Intern2/Code/prepare-inference
  uvicorn backend.api:app --host 0.0.0.0 --port 8000

  # Tab 2 – start the UI
  streamlit run frontend/app.py
"""

import os
import time
import json
import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# Default RAG knowledge base path (can be overridden via environment variable)
# Set RAG_CSV_PATH env var to use a different location
DEFAULT_RAG_CSV = os.getenv(
    "RAG_CSV_PATH", "your path to csv file"
)

st.set_page_config(
    page_title="Khmer BankChat",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
    color: #e0e0ff;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stSidebar"] * { color: #d0d0ff !important; }

  /* Chat messages */
  [data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(10px);
  }

  /* Chat input */
  [data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(120,100,255,0.4) !important;
    border-radius: 12px !important;
    color: #e0e0ff !important;
    font-family: 'Inter', sans-serif !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
  }

  /* Status badge */
  .status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px 0;
  }
  .badge-green { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #34d399; }
  .badge-red   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid #f87171; }
  .badge-blue  { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid #818cf8; }

  /* Section headers */
  .section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #818cf8;
    margin: 18px 0 8px 0;
  }

  /* Model cards */
  .model-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .model-card:hover { border-color: #7c3aed; background: rgba(124,58,237,0.1); }
  .model-card.active { border-color: #4f46e5; background: rgba(79,70,229,0.15); }

  /* Header hero */
  .hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
  }
  .hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 6px;
  }

  /* Info panels */
  .info-panel {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
  }

  /* Spinner override */
  [data-testid="stSpinner"] { color: #818cf8 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

  /* RAG hits */
  .rag-hit {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #34d399;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 4px 0;
    font-size: 12px;
    color: #94a3b8;
  }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False
if "rag_num_pairs" not in st.session_state:
    st.session_state.rag_num_pairs = 0
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful Khmer banking assistant. "
        "Always answer in Khmer unless the user writes in another language. "
        "Be concise, accurate, and friendly."
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:1.4rem;">🏦 BankChat</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub" style="font-size:0.8rem;">Khmer Banking Assistant</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Model Selector ──
    st.markdown(
        '<div class="section-title">⚡ Model Selection</div>', unsafe_allow_html=True
    )

    model_choice = st.radio(
        "Select Adapter",
        options=["qwen", "seallm"],
        format_func=lambda x: {
            "qwen": "🟦 Qwen3-4B  (QLoRA)",
            "seallm": "🟩 SeaLLMs-v3-7B (QLoRA)",
        }[x],
        key="model_radio",
        label_visibility="collapsed",
    )

    quant_choice = st.selectbox(
        "Quantization",
        ["4bit", "8bit", "none"],
        index=0,
        help="4-bit saves the most VRAM; 'none' uses full precision",
    )

    if st.button("🚀 Load Selected Model", use_container_width=True):
        with st.spinner(
            f"Loading {'Qwen3' if model_choice == 'qwen' else 'SeaLLMs'} …"
        ):
            try:
                r = requests.post(
                    f"{API_BASE}/load_model",
                    json={"model_id": model_choice, "quantization": quant_choice},
                    timeout=300,
                )
                if r.ok:
                    st.session_state.loaded_model = model_choice
                    st.success(f"✅ Model loaded!")
                    st.session_state.messages = []  # reset history on model switch

                    # Auto-load RAG knowledge base if not already loaded
                    if not st.session_state.rag_loaded:
                        with st.spinner("Loading knowledge base …"):
                            try:
                                rag_r = requests.post(
                                    f"{API_BASE}/rag/load",
                                    json={"csv_path": DEFAULT_RAG_CSV, "top_k": 3},
                                    timeout=180,
                                )
                                if rag_r.ok:
                                    data = rag_r.json()
                                    st.session_state.rag_loaded = True
                                    st.session_state.rag_num_pairs = data.get(
                                        "num_pairs", 0
                                    )
                                    st.success(
                                        f"✅ RAG: {data['num_pairs']:,} Q&A pairs indexed!"
                                    )
                            except Exception:
                                pass  # RAG is optional, don't block on failure
                else:
                    st.error(f"Error: {r.json().get('detail', r.text)}")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")

    # Current model badge
    if st.session_state.loaded_model:
        label = {"qwen": "Qwen3-4B", "seallm": "SeaLLMs-v3-7B"}[
            st.session_state.loaded_model
        ]
        st.markdown(
            f'<span class="status-badge badge-green">✓ {label} active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-red">✗ No model loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── RAG Section ──
    st.markdown(
        '<div class="section-title">📚 RAG Knowledge Base</div>', unsafe_allow_html=True
    )

    rag_top_k = st.slider("Top-K retrieved pairs", 1, 10, 3)

    col_rag1, col_rag2 = st.columns(2)
    with col_rag1:
        if st.button("Load KB", use_container_width=True):
            with st.spinner("Building FAISS index …"):
                try:
                    r = requests.post(
                        f"{API_BASE}/rag/load",
                        json={"csv_path": DEFAULT_RAG_CSV, "top_k": rag_top_k},
                        timeout=180,
                    )
                    if r.ok:
                        data = r.json()
                        st.session_state.rag_loaded = True
                        st.session_state.rag_num_pairs = data.get("num_pairs", 0)
                        st.success(f"✅ {data['num_pairs']:,} Q&A pairs indexed!")
                    else:
                        st.error(r.json().get("detail", r.text))
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_rag2:
        use_rag = st.toggle("Use RAG", value=st.session_state.rag_loaded)

    if st.session_state.rag_loaded:
        st.markdown(
            f'<span class="status-badge badge-blue">📚 {st.session_state.rag_num_pairs:,} pairs</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Generation Settings ──
    st.markdown(
        '<div class="section-title">⚙️ Generation Settings</div>', unsafe_allow_html=True
    )

    max_new_tokens = st.slider("Max new tokens", 64, 2048, 512, step=64)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, step=0.05)
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, step=0.05)
    streaming = st.toggle("Streaming output", value=True)

    st.divider()

    # ── System prompt ──
    st.markdown(
        '<div class="section-title">🤖 System Prompt</div>', unsafe_allow_html=True
    )
    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=100,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Utility buttons ──
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # ── API status ──
    st.markdown(
        '<div class="section-title">🌐 API Status</div>', unsafe_allow_html=True
    )
    try:
        status = requests.get(f"{API_BASE}/status", timeout=3).json()
        m = status.get("model", {})
        r = status.get("rag", {})
        st.markdown(
            f'<div class="info-panel">🖥️ GPU: {"✅ " + str(m.get("gpu_count", 0)) + "x" if m.get("cuda_available") else "❌ CPU"}<br>'
            f"🤖 Model: {m.get('label', '—')}<br>"
            f"📚 RAG pairs: {r.get('num_pairs', 0):,}<br>"
            f"🔤 Khmer seg: {'✅' if r.get('khmer_segmentation') else '❌'}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<span class="status-badge badge-red">API offline</span>',
            unsafe_allow_html=True,
        )


# ── Main area ───────────────────────────────────────────────────────────────
# Header
st.markdown(
    '<div class="hero-title">🏦 Khmer BankChat</div>'
    '<div class="hero-sub">AI-powered Khmer Banking Assistant with RAG · Switch between Qwen3 & SeaLLMs adapters</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rag_hits"):
            with st.expander("📚 Retrieved context snippets"):
                for hit in msg["rag_hits"]:
                    dist = hit.get("distance", 0)
                    st.markdown(
                        f'<div class="rag-hit"><b>Q:</b> {hit["question"][:80]}…<br>'
                        f"<b>A:</b> {hit['answer'][:100]}… <i>(dist: {dist:.3f})</i></div>",
                        unsafe_allow_html=True,
                    )

# Chat input
placeholder_text = "សួរអំពីគណនី ប័ណ្ណ ឬប្រតិបត្តិការ…  (ask anything in Khmer or English)"
if prompt := st.chat_input(
    placeholder_text, disabled=st.session_state.loaded_model is None
):
    # ── Display user message ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Build API messages ──
    api_messages = [{"role": "system", "content": st.session_state.system_prompt}]
    # Include last 10 turns for context
    for m in st.session_state.messages[-20:]:
        if m["role"] in ("user", "assistant"):
            api_messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "messages": api_messages,
        "use_rag": use_rag and st.session_state.rag_loaded,
        "rag_top_k": rag_top_k,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": rep_penalty,
        "stream": streaming,
    }

    # ── Generate response ──
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        rag_hits = []

        try:
            if streaming:
                endpoint = f"{API_BASE}/chat/stream"
                with requests.post(
                    endpoint, json=payload, stream=True, timeout=300
                ) as resp:
                    for line in resp.iter_lines():
                        if line and line.startswith(b"data: "):
                            data = json.loads(line[6:])
                            if "error" in data:
                                st.error(data["error"])
                                break
                            full_response += data.get("token", "")
                            response_placeholder.markdown(full_response + "▋")
            else:
                endpoint = f"{API_BASE}/chat"
                resp = requests.post(endpoint, json=payload, timeout=300)
                if resp.ok:
                    full_response = resp.json().get("answer", "")
                else:
                    st.error(resp.json().get("detail", resp.text))

            response_placeholder.markdown(full_response)

            # Fetch RAG hits for display (optional second call, cheap)
            if use_rag and st.session_state.rag_loaded:
                try:
                    hits_resp = requests.post(
                        f"{API_BASE}/rag/retrieve",
                        json={"query": prompt, "top_k": rag_top_k},
                        timeout=10,
                    )
                    if hits_resp.ok:
                        rag_hits = hits_resp.json().get("hits", [])
                except Exception:
                    pass  # non-critical

            if rag_hits:
                with st.expander("📚 Retrieved context snippets"):
                    for hit in rag_hits:
                        dist = hit.get("distance", 0)
                        st.markdown(
                            f'<div class="rag-hit"><b>Q:</b> {hit["question"][:80]}…<br>'
                            f"<b>A:</b> {hit['answer'][:100]}… <i>(dist: {dist:.3f})</i></div>",
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            full_response = f"[Error: {e}]"

    # ── Save assistant turn ──
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "rag_hits": rag_hits,
        }
    )

# Show hint when no model is loaded
if st.session_state.loaded_model is None:
    st.info(
        "👈 **Select a model in the sidebar and click 'Load Selected Model' to start chatting.**"
    )
