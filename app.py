import streamlit as st
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- 1. Load docs
@st.cache_resource
def load_docs(path="docs"):
    loader = DirectoryLoader(path, loader_cls=TextLoader)
    docs = loader.load()
    return docs

# --- 2. Chunk docs
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# --- 3. Create Vector DB
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = OllamaEmbeddings(model="llama3")  # You can use mistral, gemma, etc.
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_ollama")
    return vectordb

# --- 4. RAG Chain using Ollama LLM
def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = Ollama(model="llama3")  # Change model if needed
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI
st.set_page_config(page_title="📘 Network SOP Assistant", layout="wide")
st.title("📘RAG-Powered Network SOP Assistant")

# --- Load + Embed
docs = load_docs()
chunks = chunk_docs(docs)
vectordb = create_vectorstore(chunks)
qa_chain = get_qa_chain(vectordb)

# --- Ask
query = st.text_input("Ask a question from your docs")
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
    st.success(response)

# --- Optional: show docs
with st.expander("📄 Show Raw Docs"):
    for doc in docs:
        st.write(doc.metadata["source"])
        st.code(doc.page_content[:500] + "...", language="markdown")
