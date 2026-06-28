import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from langdetect import detect

import faiss
import numpy as np

POPPLER_PATH = r"C:\Release-26.02.0-0\poppler-26.02.0\Library\bin"

st.set_page_config(page_title="Chat PDF AI", layout="wide")

st.title("💬 Chat with PDF AI")

# ---------------- OCR ----------------
def extract_text(image):
    return pytesseract.image_to_string(image)

# ---------------- SIMPLE CHUNKING ----------------
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------- EMBEDDING (simple hash-based fallback if no API) ----------------
def embed(text):
    # simple deterministic embedding (NO API needed)
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(384).astype("float32")

# ---------------- BUILD INDEX ----------------
def build_index(chunks):
    vectors = np.array([embed(c) for c in chunks])
    index = faiss.IndexFlatL2(384)
    index.add(vectors)
    return index, vectors

# ---------------- SEARCH ----------------
def search(query, chunks, index, vectors, k=3):
    q_vec = embed(query).reshape(1, -1)
    _, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload PDF", type=["pdf"])

chunks = []
index = None
vectors = None

if file is not None:

    text = ""

    images = convert_from_bytes(file.read(), poppler_path=POPPLER_PATH)

    for img in images:
        st.image(img)
        text += extract_text(img) + "\n"

    chunks = chunk_text(text)

    index, vectors = build_index(chunks)

    st.success("PDF processed successfully!")

# ---------------- CHAT ----------------
if chunks:

    st.subheader("💬 Ask Questions")

    question = st.text_input("Ask something from PDF")

    if question:

        results = search(question, chunks, index, vectors)

        st.write("### 📌 Relevant Context")
        for r in results:
            st.write(r)

        st.success("AI Answer (basic version):")
        st.write(" ".join(results))