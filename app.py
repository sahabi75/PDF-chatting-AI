import streamlit as st
from dotenv import load_dotenv
from utils import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

load_dotenv()
st.set_page_config(page_title="PDF Chat AI", page_icon="📚")
st.title("📚 Chat with your PDF")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Upload PDFs")
    pdfs = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process") and pdfs:
        with st.spinner("Processing..."):
            text = get_pdf_text(pdfs)

            st.write("TEXT LENGTH:", len(text))

            chunks = get_text_chunks(text)

            st.write("CHUNKS:", len(chunks))

            vs = get_vectorstore(chunks)

            st.session_state.chain = get_conversation_chain(vs)

            st.success("Ready to chat!")

question = st.chat_input("Ask something about your PDF...")
if question and st.session_state.chain:
    response = st.session_state.chain({"question": question})
    st.session_state.history.append(("You", question))
    st.session_state.history.append(("AI", response["answer"]))

for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.write(msg)