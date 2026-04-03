import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text) 
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatting AI", page_icon=":books:", layout="centered")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.title("PDF Chatting AI")
    st.text_input("Upload a PDF file and start chatting with it!")
    
    st.write(user_template.replace("{{MSG}}", "Hello! PDF AI."), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello! Please upload a PDF file to start chatting."), unsafe_allow_html=True)
    with st.sidebar:
         st.subheader("Upload PDF")
         pdf_docs = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)    
         if st.button("Process PDF"):
             with st.spinner("Processing..."):
                 raw_text = get_pdf_text(pdf_docs)
                 
                 # chunking the text
                 text_chunks = get_text_chunks(raw_text) 
                 st.write(text_chunks)  
                 # vectorstore 
                 vectorstore = get_vectorstore(text_chunks)
                 
                 #conversation chain
                 st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                 
    st.session_state.conversation 
    
                 
                      
         

if __name__ == "__main__":     main()