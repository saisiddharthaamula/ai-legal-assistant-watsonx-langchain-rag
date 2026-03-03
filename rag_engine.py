import os
import re
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

DATA_PATH = "data"
INDEX_PATH = "embeddings/faiss.index"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    # Remove page numbers like "Page 6 of 22"
    text = re.sub(r'Page \d+ of \d+', '', text)

    # Remove standalone numbers
    text = re.sub(r'\n\d+\n', '\n', text)

    # Fix common broken OCR words
    text = re.sub(r'\bich\b', 'which', text)

    # Remove extra line breaks
    text = re.sub(r'\n+', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# LOAD PDF DOCUMENTS
# -----------------------------
def load_documents():
    documents = []
    sources = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_PATH, file))
            text = ""

            for page in reader.pages:
                extracted = page.extract_text() or ""
                text += extracted

            text = clean_text(text)

            documents.append(text)
            sources.append(file)

    return documents, sources


# -----------------------------
# CHUNK TEXT (Sentence Based)
# -----------------------------
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------
# BUILD FAISS INDEX
# -----------------------------
def build_index():

    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    documents, sources = load_documents()

    all_chunks = []
    chunk_sources = []

    for doc, source in zip(documents, sources):
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        chunk_sources.extend([source] * len(chunks))

    embeddings = model.encode(all_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    return index, all_chunks, chunk_sources


# -----------------------------
# LOAD INDEX
# -----------------------------
def load_index():
    if not os.path.exists(INDEX_PATH):
        return build_index()

    index = faiss.read_index(INDEX_PATH)
    documents, sources = load_documents()

    all_chunks = []
    chunk_sources = []

    for doc, source in zip(documents, sources):
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        chunk_sources.extend([source] * len(chunks))

    return index, all_chunks, chunk_sources


# -----------------------------
# RETRIEVE RELEVANT PASSAGES
# -----------------------------
def retrieve(query, top_k=3):
    index, chunks, sources = load_index()

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in I[0]:
        results.append({
            "text": chunks[idx],
            "source": sources[idx]
        })

    return results

from ibm_watsonx_ai.foundation_models import ModelInference
# -----------------------------
# GENERATE STRUCTURED ANSWER
# -----------------------------
def generate_answer(query, top_k=3):
    passages = retrieve(query, top_k)
    context = "\n\n".join([p["text"] for p in passages])

    model = ModelInference(
        model_id="ibm/granite-4-h-small",  # Updated model
        credentials={
            "apikey": os.getenv("WATSON_API_KEY"),
            "url": os.getenv("WATSON_URL")
        },
        project_id=os.getenv("WATSON_PROJECT_ID")
    )

    prompt = f"""
You are a professional legal assistant.

Use ONLY the legal context provided below.
Do NOT make assumptions.
If the answer is not clearly supported by the context, explicitly say so.

------------------------
CONTEXT:
{context}
------------------------

QUESTION:
{query}

Provide a structured answer with:

1. Legal Principle (2–3 lines)
2. Clear Explanation (concise and precise)
3. Final Conclusion (1 line)

Keep it professional, under 250 words.
Avoid repetition.
Do not copy raw paragraphs.
Summarize clearly.
"""

    response = model.generate(
        prompt=prompt,
        params={
            "max_new_tokens": 400,
            "temperature": 0.2,
            "decoding_method": "greedy"
        }
    )

    answer_text = response["results"][0]["generated_text"]

    return {
        "answer": answer_text.strip(),
        "passages": passages
    }