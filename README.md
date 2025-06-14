# Document-Summarization-using-Retrieval-Augmented-Generation-RAG-
This RAG-powered document summarizer uses FAISS for semantic search and BART (HuggingFace) for abstractive summarization. It processes arXiv/PDF/TXT with smart chunking, generates technical/novelty/applications summaries, and evaluates with ROUGE/BERTScore. GPU-accelerated via PyTorch, it supports custom queries and handles long documents.

# Retrieval-Augmented Generation (RAG) Based Question Answering System

## Table of Contents
1. [Introduction to RAG](#introduction-to-rag)
2. [Project Overview](#project-overview)
3. [Code Explanation](#code-explanation)
4. [Why This Approach?](#why-this-approach)
5. [Setup & Usage](#setup--usage)

---

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a hybrid architecture that combines the power of **retrieval-based methods** with **large language models (LLMs)** to generate responses grounded in external data sources. Unlike traditional LLMs that rely solely on pretraining, RAG enhances accuracy, factuality, and transparency by retrieving relevant documents at runtime and incorporating them into the generation process.

---

## Project Overview

This project is a complete implementation of a RAG pipeline designed to:
- Load and embed knowledge sources (PDFs, text files, etc.)
- Retrieve the most relevant context for any user query
- Use a language model to generate accurate, context-aware answers

Built using **LangChain**, **FAISS/Chroma**, and **Hugging Face or OpenAI models**, this pipeline is modular and easy to customize for domain-specific applications like enterprise search, document Q&A, or internal chatbots.

---

## Code Explanation

The code is organized into three main stages that reflect the RAG pipeline:

### 1.Document Ingestion & Embedding
- Loads documents from local directories using LangChain loaders.
- Splits content into chunks using `RecursiveCharacterTextSplitter` for improved context handling.
- Embeds text using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) or OpenAI’s embedding API.
- Stores vectors in a fast and scalable vector database (FAISS or ChromaDB).

> **Why**: Converts text into a format that allows for semantic search. Enables quick, meaningful retrieval based on the intent of queries rather than keywords.
> **Why?** Handle diverse input formats consistently
- *Implementation*:
  - PDF text extraction via PyPDF2
  - LaTeX cleaning with regex patterns
  - Metadata preservation (titles/authors/sections)

---

### 2.Query-Time Retrieval
- A user query is embedded similarly to the documents.
- The system performs a similarity search in the vector store to find the top-matching document chunks.
- Uses retrievers like LangChain's `VectorStoreRetriever` to ensure high-relevance matches.

> **Why**: Ensures the model only sees content relevant to the user query, improving answer quality and reducing hallucinations.
- **Why?** Balance speed and accuracy
- *Implementation*:
  - FAISS HNSW index (hierarchical graph search)
  - Sentence-BERT embeddings (all-mpnet-base-v2)
  - Hybrid scoring (content + metadata boosting)
---

### 3.Context-Aware Generation
- Uses `RetrievalQA` or `ConversationalRetrievalChain` to wrap an LLM.
- The model is provided both the user query and the retrieved context.
- Returns a grounded answer, often with references to source material.

> **Why**: Fuses factual retrieval with fluent generation. Ensures responses are based on known sources, boosting reliability and explainability.
> **Summary Generator**
- *Why?* Produce fluent, focused summaries
- *Implementation*:
  - BART-large-CNN (seq2seq architecture)
  - Beam search (4 beams) with length penalty
  - Focus mode conditioning


---

## Why This Approach?

- **Accuracy**: Answers are generated based on actual documents.
- **Reduced Hallucination**: Model avoids making up information.
- **Flexible**: Add/remove documents without retraining.
- **Modular**: Easy to swap components (embeddings, LLMs, retrievers).
- **Explainable**: Retrieved context can be shown for transparency.

---

## Setup & Usage

### Installation
```bash
git clone https://github.com/your-username/your-rag-project.git
cd your-rag-project
pip install -r requirements.txt
---

### Installation - 2 
```bash
conda create -n rag python=3.8
conda activate rag
conda install -c conda-forge faiss-gpu pytorch=2.0
pip install -r requirements.txt

