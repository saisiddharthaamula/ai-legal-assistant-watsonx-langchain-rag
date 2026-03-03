AI Legal Assistant (WatsonX + LangChain RAG)
📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) based AI Legal Assistant for semantic legal document analysis. It retrieves contextually relevant legal passages using vector embeddings and generates structured, source-grounded responses.

🚀 Key Features

Semantic Search using Sentence Transformers

FAISS Vector Similarity Indexing

Structured Legal Output (Principle, Explanation, Conclusion)

Source Attribution for Transparency

Streamlit Web Interface

Optional IBM WatsonX Integration

🏗 Architecture

Legal PDF
→ Text Extraction
→ Chunking
→ Embedding Generation
→ FAISS Index
→ Query Embedding
→ Top-k Retrieval
→ Structured Response

🛠 Technology Stack

Python 3.10

Sentence Transformers

FAISS

LangChain

IBM WatsonX

Streamlit

⚙ Installation
git clone https://github.com/your-username/AI-Legal-Assistant-WatsonX-LangChain-RAG.git
cd AI-Legal-Assistant-WatsonX-LangChain-RAG
pip install -r requirements.txt
streamlit run src/streamlit_app.py
⚖ Ethical Disclaimer

This system is developed strictly for academic research purposes and does not provide legal advice.
