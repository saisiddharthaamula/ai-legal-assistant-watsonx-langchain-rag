import streamlit as st
from rag_engine import generate_answer
import os
import streamlit as st
from rag_engine import generate_answer, build_index

st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️")

st.title("⚖️ AI Legal Assistant (Watson + LangChain RAG)")

st.markdown(
"⚠️ *This system is for academic and informational purposes only. It does not constitute legal advice.*"
)

st.write("Ask any legal question based on uploaded court judgments.")

st.subheader("📂 Upload Court Judgments")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("Documents uploaded successfully!")

    # Rebuild FAISS index
    with st.spinner("Rebuilding search index..."):
        build_index()

    st.success("Index updated successfully!")

query = st.text_area("Enter your legal question:", height=150)

top_k = st.slider("Number of passages to retrieve:", 1, 5, 3)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving legal context..."):
            result = generate_answer(query, top_k)

        st.subheader("Short Answer")
        st.write(result["answer"])

        st.subheader("Supporting Legal Extracts")

        for p in result["passages"]:
            st.markdown(f"**Source:** {p['source']}")
            st.write(p["text"])
            st.write("---")