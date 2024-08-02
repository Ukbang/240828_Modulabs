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

# @st.cache_data(show_spinner="답변 생성 중...")
@st.cache_resource(show_spinner="Loading...")
def embed_file(file):

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600, # 나눌 문자는 600개
    chunk_overlap=100, # 겹칠 문자는 100개
    separator='\n',)

    loader = UnstructuredFileLoader(f"pages/.cache/{file}")

    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(f"pages/.cache/{file}")

    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, cache_dir)

    vector_store = FAISS.from_documents(docs, cached_embeddings)

    retriever = vector_store.as_retriever()

    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

def paint_history():
    if len(st.session_state["messages"]) <= 20:
        for message in st.session_state["messages"]:
            send_message(message["message"], message["role"], save=False)
    else:
        for message in st.session_state["messages"][-20:]:
            send_message(message["message"], message["role"], save=False)        

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *arg, **kwargs):
        st.session_state["messages"].append({"message":self.message, "role":"ai"})

    def on_llm_new_token(self, token, *args, **klwargs):
        self.message += token
        self.message_box.markdown(self.message)

chat = ChatOpenAI(
    temperature=0.1,
    # model="gpt-4-0125-preview",
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    다음 문맥만을 사용하여 문제에 답하세요. 답을 모른다면 이 문서에서는 해당 내용을 찾을 수 없다고 말하세요. 답변을 지어내지 마세요.
    그리고 대답은 항상 한글로 대답해주세요. Always Answer Only Korean.

     context : {context}
    """),
    ("human", "{question}")
])

st.title("Document GPT")

st.markdown("""
##### 환영합니다!
            
**왼쪽 사이드바에 여러분의 파일을 업로드 해주세요.**
            
**어느 것이던 물어보세요!**
""")
with st.sidebar:
    file = st.file_uploader("txt파일 혹은 pdf파일을 업로드하세요.", type=["pdf", "txt"])

if file:
    file_content = file.read()
    file_path = f"pages/.cache/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    retriever = embed_file(file.name)

    send_message("무엇이든 물어보세요!", "ai", save=False)
    paint_history()
    message = st.chat_input("파일에 대해 물어보세요.")

    if message:
        send_message(message, "human")
        chain = {
            "context":retriever | RunnableLambda(format_docs), 
            "question":RunnablePassthrough()
        } | prompt | chat

        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []