# Khmer Banking LLM — Fine-tuning & RAG Pipeline

A comprehensive internship project focused on developing production-ready Large Language Models for Khmer banking Q&A, featuring fine-tuned adapters, RAG integration, and an interactive inference interface.

## Overview

This repository contains the complete pipeline for training, evaluating, and deploying domain-specific LLMs for Khmer language banking applications. The project demonstrates end-to-end machine learning engineering from data preparation through production deployment.

### Key Achievements

- Fine-tuned multiple LLM architectures (Qwen, SeaLLMs, Gemma) using QLoRA
- Implemented RAG with Khmer-specific tokenization and semantic search
- Built production-ready inference API with streaming support
- Developed comprehensive model evaluation frameworks

---

## Project Structure

```
Code/
├── prepare-inference/       # Production inference system (FastAPI + Streamlit)
│   ├── backend/             # API server, model management, RAG engine
│   ├── frontend/            # Streamlit chat interface
│   ├── qwen-model-ok/       # Fine-tuned Qwen3-4B adapter
│   └── SeaLLM-model-ok/     # Fine-tuned SeaLLMs-v3-7B adapter
│
├── Qwen/                    # Qwen model training experiments
│   ├── train.ipynb          # Main training pipeline
│   ├── model_evaluation.ipynb
│   └── model-done/          # Final trained checkpoints
│
├── Qwen3.5/                 # Latest Qwen training iterations
├── SeaLLm/                  # SeaLLMs fine-tuning & evaluation
├── gamma/                   # Gemma model experiments
├── New-Gamma/               # Gemma with Khmer RAG segmentation

```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Base Models** | Qwen3-4B, SeaLLMs-v3-7B, Gemma-3-4B |
| **Fine-tuning** | QLoRA (4-bit quantization), PEFT, Unsloth |
| **RAG** | FAISS, Sentence-Transformers, KhmerNLTK |
| **Backend** | FastAPI, Uvicorn, BitsAndBytes |
| **Frontend** | Streamlit |
| **Evaluation** | Custom metrics, Grounding Score |

---

## Model Fine-tuning

### Approach

All models were fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) for efficient training on consumer GPUs:

| Model | Base | Adapter Size | LoRA Rank | Use Case |
|-------|------|--------------|-----------|----------|
| Qwen | `unsloth/qwen3-4b-unsloth-bnb-4bit` | ~126 MB | r=16 | Fast, lightweight reasoning |
| SeaLLM | `SeaLLMs/SeaLLMs-v3-7B-Chat` | ~308 MB | r=64 | High-quality Khmer NLP |
| Gemma | `google/gemma-3-4b` | ~100 MB | r=16 | Experimental comparisons |

### Training Highlights

- **Dataset**: Khmer banking Q&A pairs
- **Quantization**: 4-bit NF4 with double quantization
- **Optimizer**: AdamW with cosine learning rate schedule
- **Hardware**: CUDA-capable GPU (T4, VRAM16GB)

---

## RAG Implementation

The Retrieval-Augmented Generation system enhances model responses with relevant banking knowledge:

### Features

- **Khmer Tokenization**: Uses KhmerNLTK for proper word segmentation
- **Semantic Search**: Multilingual sentence embeddings (`paraphrase-multilingual-mpnet-base-v2`)
- **FAISS Indexing**: Fast similarity search across knowledge base
- **Context Injection**: RAG context injected into latest user message only

### Architecture

```
User Query → Khmer Segmentation → Embedding → FAISS Search
                                                    ↓
                                           Top-K Q&A Pairs
                                                    ↓
                              Context Injection → LLM Generation
```

---

## Inference System

The `prepare-inference/` directory contains a production-ready chat interface:

### Features

- **Dynamic Model Switching**: Hot-swap between Qwen and SeaLLM adapters
- **Streaming Responses**: Server-Sent Events (SSE) for real-time token streaming
- **Conversational Memory**: Maintains context across multi-turn conversations (last 20 messages)
- **Thread-Safe Generation**: Singleton model manager prevents memory corruption

### Quick Start

```bash
cd prepare-inference

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start services
./start_all.sh

# Access UI at http://localhost:8501
# API docs at http://localhost:8000/docs
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/adapters` | GET | List loaded model adapters |
| `/status` | GET | Current model & hardware status |
| `/load_model` | POST | Switch active adapter |
| `/chat` | POST | Synchronous text generation |
| `/chat/stream` | POST | SSE streaming generation |

---

## Model Evaluation

Comprehensive evaluation framework for comparing model performance:

### Metrics

- **Grounding Score**: Measures factual accuracy against knowledge base
- **Response Quality**: Semantic similarity to reference answers
- **Latency**: Token generation speed

### Results

Evaluation results available in:
- `quick_rag_evaluation_results.csv`
- `SeaLLm/evaluation_results_seallm_detailed.csv`
- `plot/` directory for visualizations

---

## Development Notebooks

| Notebook | Purpose |
|----------|---------|
| `Qwen/train.ipynb` | Main Qwen fine-tuning pipeline |
| `SeaLLm/seallm_qlora_finetune_kaggle.ipynb` | SeaLLM QLoRA training |
| `Qwen/model_evaluation.ipynb` | Evaluation framework |
| `New-Gamma/khmer-rag-segmentation.ipynb` | Khmer tokenization experiments |
| `plot/result.ipynb` | Visualization & analysis |

---

## Requirements

### Hardware
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM

### Software
- Python 3.10+
- CUDA 11.8+ with cuDNN
- Key packages: `transformers`, `peft`, `bitsandbytes`, `unsloth`, `faiss-cpu`, `khmernltk`

---

## Future Work

- [ ] Expand Khmer banking dataset coverage
- [ ] Implement quantization for edge deployment
- [ ] Add multi-model ensemble inference
- [ ] Develop automated evaluation pipeline

---

## Acknowledgments

This project was developed as part of an internship program, focusing on advancing Khmer language AI capabilities in the financial domain.

---

## Author

**JAHAD69**

---

## License

This project is for educational and research purposes.
