from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"}
)

def create_vectorstore(chunks, metadatas):
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadatas
    )
    return vectorstore

def save_vectorstore(vectorstore):
    vectorstore.save_local("vectorstore")

def load_vectorstore():
    return FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )