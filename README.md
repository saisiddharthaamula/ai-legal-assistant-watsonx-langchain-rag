Title

AI Legal Assistant: A Retrieval-Augmented Generation Framework for Legal Question Answering using IBM Watsonx

Abstract

The legal domain involves extensive documentation and complex reasoning, making information retrieval and interpretation challenging. This project proposes an AI-powered Legal Assistant using a Retrieval-Augmented Generation (RAG) framework. The system integrates semantic search via FAISS and sentence embeddings with IBM Watsonx foundation models to generate structured legal answers. Court judgments are processed into semantically meaningful chunks, embedded into vector space, and indexed for similarity-based retrieval. Upon receiving a query, the system retrieves relevant legal passages and generates contextualized legal explanations. The proposed solution demonstrates effective integration of vector databases and large language models for domain-specific question answering.

Keywords: Legal AI, Retrieval-Augmented Generation, FAISS, IBM Watsonx, NLP, Semantic Search

I. Introduction

Legal research requires analyzing large volumes of case law and statutory documents. Traditional keyword-based search systems often fail to capture contextual meaning.

Recent advances in Large Language Models (LLMs) enable contextual understanding and generative capabilities. However, standalone LLMs may hallucinate or produce unsupported claims. Retrieval-Augmented Generation (RAG) addresses this limitation by grounding responses in retrieved documents.

This project develops an AI Legal Assistant combining:

Semantic document retrieval

Vector database indexing

Foundation model-based generation

Interactive user interface

II. Literature Survey

Recent research in NLP has demonstrated:

Effectiveness of transformer-based embeddings for semantic similarity

Vector databases for scalable document retrieval

RAG frameworks to reduce hallucinations

Domain-specific LLM applications in legal AI

Studies indicate that combining retrieval systems with generative models improves factual grounding and response reliability.

III. System Architecture
A. Overall Architecture

User Query
↓
Embedding Generation
↓
FAISS Vector Search
↓
Top-K Relevant Chunks
↓
IBM Watsonx LLM
↓
Structured Legal Response

B. Module Description
1. Document Ingestion Module

Accepts court judgment PDFs

Extracts text using PyPDF

Performs text cleaning

2. Text Chunking Module

Splits documents into 500-character segments

Ensures semantic continuity

3. Embedding Module

Uses SentenceTransformer (all-MiniLM-L6-v2)

Converts text chunks into 384-dimensional vectors

4. Vector Database Module

Uses FAISS IndexFlatL2

Stores embeddings for similarity search

5. Retrieval Module

Encodes user query

Retrieves Top-K semantically similar chunks

6. Generation Module

Integrates IBM Watsonx Granite model

Generates structured legal explanation

7. User Interface

Built using Streamlit

Supports PDF upload and interactive queries

IV. Methodology
Step 1: Data Collection

Public domain court judgments were used.

Step 2: Preprocessing

PDF text extraction

Noise removal

Text normalization

Step 3: Embedding and Indexing

Each text chunk was embedded using sentence transformers and indexed using FAISS.

Step 4: Query Processing

User queries were embedded into vector space.

Step 5: Retrieval

Top-K nearest neighbors retrieved via L2 distance.

Step 6: Response Generation

Retrieved passages were combined and passed to IBM Watsonx for structured response generation.

V. Experimental Results
A. Evaluation Criteria

Relevance of retrieved passages

Contextual accuracy

Coherence of generated answers

Reduction in hallucinated responses

B. Observations

FAISS retrieval significantly improved factual grounding

Structured answer format improved clarity

IBM Watsonx provided domain-consistent legal reasoning

VI. Advantages of Proposed System

Reduces hallucination compared to standalone LLM

Efficient semantic search

Scalable architecture

Dynamic document upload support

Interactive web interface

VII. Limitations

Dependent on document quality

Performance decreases with very large datasets without optimization

Requires API access for Watsonx

VIII. Future Scope

Citation extraction automation

Multi-case comparative analysis

Role-based explanation modes

Cloud deployment

Legal precedent ranking system

Integration with statutory databases

IX. Conclusion

The AI Legal Assistant demonstrates a practical implementation of Retrieval-Augmented Generation in the legal domain. By combining semantic search with IBM Watsonx foundation models, the system provides context-aware and grounded legal answers. The framework illustrates how vector databases and large language models can be effectively integrated for domain-specific AI solutions.

References (Sample IEEE Format)

[1] T. Brown et al., “Language Models are Few-Shot Learners,” Advances in Neural Information Processing Systems, 2020.

[2] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with FAISS,” IEEE Transactions on Big Data, 2019.

[3] IBM, “Watsonx AI Documentation,” IBM Cloud Documentation, 2024.