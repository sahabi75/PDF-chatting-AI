from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    print("TEXT SIZE =", len(text))

    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    print("TOTAL CHUNKS =", len(chunks))

    return chunks


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt_template = """
You are a helpful AI assistant that answers questions about the uploaded PDF.

Use the following pieces of context from the PDF to answer the user's question.

Context:
{context}

Question:
{question}

Instructions:
- Answer based on the PDF context whenever possible.
- If the answer is not available in the PDF, clearly say that the information is not available in the uploaded PDF.
- Do not make up information.
- Be clear and concise.

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt
        }
    )

    return chain
