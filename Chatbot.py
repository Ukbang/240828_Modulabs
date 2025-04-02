import streamlit as st
import os
import dotenv
from typing import Optional
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

dotenv.load_dotenv()

# os.environ["openai_api_key"] = "Your API-Key"

st.set_page_config(page_title="모두해유",
                   page_icon="🤖")

st.title("모두해유 고객 응대 챗봇 만들기!")

st.markdown("""\n
모두해유 세미나에 오신것을 환영합니다!\n
왼쪽 사이드바에서 파일을 선택해주세요.
""")

# 파일 임베딩 수행
@st.cache_resource(show_spinner="Loading...")
def embedding_file(file: str) -> VectorStoreRetriever:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n"],
    )

    loader = PyPDFLoader(f"./files/{file}")
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()    

    return retriever

# 파일 선택
with st.sidebar:
    chat_clear = None
    model_name = st.selectbox(label="모델을 선택해주세요.", placeholder= "Select Your Model", options=["gpt-3.5-turbo-0125", "gpt-4o-mini"], index=None)
    file = st.selectbox(label="파일을 선택해주세요.", placeholder= "Select Your File", options=(os.listdir("./files")), index=None)
    if file:
        retriever = embedding_file(file)
        st.success(f"임베딩이 완료되었습니다.")
        st.info(f"현재 임베딩 된 파일명 : \n\n **{file}**")
        col1, col2 = st.columns(2)
        chat_clear = col1.button("대화 내용 초기화")
        embed_clear = col2.button("임베딩 초기화")

        llm = ChatOpenAI(temperature=0.1,
                        model=model_name)          

        if embed_clear:
            st.cache_resource.clear()

if "history" not in st.session_state or chat_clear:
    st.session_state["history"] = []    

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    context : {context}

    당신은 언제나 고객에게 최선을 다해 답변을 하며 말투는 굉장히 친근합니다. 직업은 전문 상담원입니다. 답변 시, 아래의 규칙을 지켜야만 합니다.
    규칙 1. 주어진 context만을 이용하여 답변해야합니다. 
    규칙 2. 주어진 context에서 답변을 할 수 없다면 "해당 문의는 010-2255-3366으로 연락주시면 도와드리겠습니다. 영업 시간은 오전 10시-오후 6시입니다." 라고 대답하세요.
    규칙 3. 문자열에 A1, A2, A11, A22 등 필요 없는 문자는 제거한 뒤 출력합니다.
    규칙 4. 항상 친절한 말투로 응대합니다.
    규칙 5. 웹사이트 링크를 그대로 출력합니다. 대소문자를 명확하게 구분하세요.
    """),
    ("human", "{query}")
])

query = st.chat_input("질문을 입력해주세요.")

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def chat_history(message: Optional[str] = None, role: Optional[str] = None, show: bool=True) -> None:
    if message and role:
        st.session_state["history"].append({"message":message, "role":role})   
    if show:
        for m in st.session_state["history"]:
            with st.chat_message(m["role"]):
                st.write(m["message"])

def chat_llm(query: str) -> None:
    chat_history(message=query, role="user", show=False)    
    chain = {"context":retriever | RunnableLambda(format_docs),
             "query":RunnablePassthrough()} | prompt | llm
    result = chain.invoke(query)
    chat_history(message=result.content, role = "ai")    

if query:
    chat_llm(query)
