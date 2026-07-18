import streamlit as st
from dotenv import load_dotenv

from utils import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain
)


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📚"
)


# App title
st.title("📚 Chat with your PDF")


# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None

if "history" not in st.session_state:
    st.session_state.history = []


# Sidebar
with st.sidebar:
    st.subheader("📄 Upload PDFs")

    pdfs = st.file_uploader(
        "Choose PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("⚙️ Process"):

        if not pdfs:
            st.warning("⚠️ Please upload at least one PDF first.")

        else:
            with st.spinner("📚 Processing your PDF..."):

                # Extract PDF text
                text = get_pdf_text(pdfs)

                st.write("📄 TEXT LENGTH:", len(text))

                # Create text chunks
                chunks = get_text_chunks(text)

                st.write("🧩 CHUNKS:", len(chunks))

                # Create vector store
                vectorstore = get_vectorstore(chunks)

                # Create conversation chain
                st.session_state.chain = get_conversation_chain(vectorstore)

                # Clear previous chat history
                st.session_state.history = []

                st.success("✅ PDF processed successfully!")
                st.info("💬 You can now ask questions about your PDF.")


# Chat input
question = st.chat_input("Ask something about your PDF...")


# Handle user question
if question:

    # If PDF is not processed
    if st.session_state.chain is None:

        st.warning(
            "⚠️ Please upload and process a PDF before asking questions."
        )

    else:

        with st.spinner("🤖 Thinking..."):

            response = st.session_state.chain(
                {"question": question}
            )

            answer = response["answer"]

            # Save conversation history
            st.session_state.history.append(
                ("You", question)
            )

            st.session_state.history.append(
                ("AI", answer)
            )


# Display chat history
for role, message in st.session_state.history:

    if role == "You":

        with st.chat_message("user"):
            st.write(message)

    else:

        with st.chat_message("assistant"):
            st.write(message)