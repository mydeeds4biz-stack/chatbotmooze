from io import BytesIO
import re

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


st.set_page_config(page_title="Gemini RAG Chatbot")

st.title("Gemini RAG Chatbot")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not configured. Add it in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.6-flash")

def split_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


@st.cache_data(show_spinner=False)
def extract_pdf_chunks(file_bytes, file_name):
    reader = PdfReader(BytesIO(file_bytes))
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk in split_text(page_text):
            chunks.append({"text": chunk, "source": f"{file_name}, page {page_number}"})
    return chunks


def retrieve_context(question, chunks, limit=4):
    question_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", chunk["text"].lower()))
        score = len(question_terms & chunk_terms)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:limit] if score > 0]


uploaded_files = st.file_uploader(
    "Upload PDF documents for the chatbot to use",
    type="pdf",
    accept_multiple_files=True,
)

document_chunks = []
for uploaded_file in uploaded_files or []:
    document_chunks.extend(
        extract_pdf_chunks(uploaded_file.getvalue(), uploaded_file.name)
    )

if document_chunks:
    st.caption(f"{len(document_chunks)} document chunks ready for retrieval.")
else:
    st.info("Upload at least one PDF before asking a document question.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if not document_chunks:
        answer = "Please upload a PDF document before asking a question."
    else:
        retrieved_chunks = retrieve_context(prompt, document_chunks)
        context = "\n\n".join(
            f"Source: {chunk['source']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        )
        rag_prompt = f"""Answer the question using only the document context below.
If the answer is not in the context, say that it was not found in the uploaded documents.
Include the relevant source names and page numbers when possible.

Document context:
{context}

Question:
{prompt}
"""
        response = model.generate_content(rag_prompt)
        answer = response.text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
