import streamlit as st
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS, chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="ModuhaeYou",
    page_icon="🎅"
)

st.markdown("# 안녕하세요. 무엇을 도와드릴까요?")

with st.sidebar:
    file = st.file_uploader("pdf파일을 업로드하세요.", type=["pdf"])

@st.cache_resource(show_spinner="Loading...")
def embed_file(file):

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, # 나눌 문자는 600개
    chunk_overlap=100, # 겹칠 문자는 100개
    separator='\n',)

    loader = UnstructuredFileLoader(f"../.cache/files/{file}")

    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(f"../.cache/embeddings/{file}")

    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, cache_dir)

    vector_store = FAISS.from_documents(docs, cached_embeddings)

    retriever = vector_store.as_retriever()

    return retriever