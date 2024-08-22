import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

os.environ["openai_api_key"] = "Your API-Key"

# "gpt-3.5-turbo-0125", "gpt-4o"

llm = ChatOpenAI(temperature=0.1,)

st.set_page_config(page_title="모두해유",
                   page_icon="🤖")

st.title("모두해유 고객 응대 챗봇 만들기!")

st.markdown("""\n
**왼쪽 사이드바에 여러분의 파일을 업로드 해주세요.**
            
**어느 것이던 물어보세요!**
""")

# 파일 업로더
with st.sidebar:
    file = st.file_uploader("PDF파일을 업로드해주세요.", type="pdf")

# 파일 임베딩
@st.cache_resource(show_spinner="Loading...")
def embed_file(file):

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, # 나눌 문자는 300개
    chunk_overlap=100, # 겹칠 문자는 100개
    separator='\n',)

    loader = UnstructuredFileLoader(f"./files/{file}")

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(docs, embeddings)

    retriever = vector_store.as_retriever()

    return retriever

# 파일 읽어오기
if file:
    file_content = file.read()
    file_path = f"./files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    retriever = embed_file(file.name)

def combine_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


st.session_state["history"] = []

def chat_history(message=None, role=None):
    try:
        with st.chat_message(role):     
            st.session_state["history"].append({"message":query, "role":role})
    except:
        with st.chat_message("ai"):
            st.error("오류가 발생했습니다.")
    
    for m in st.session_state["history"]:
        with st.chat_message(m["role"]):
            st.write(m["message"])




# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    당신은 주어진 context만을 이용하여 답변해야합니다. 문서에서 답을 찾을 수 없다면 "해당 문의는 제가 도와드리기 어렵습니다. 
    010-1234-1234로 연락주시면 도와드리겠습니다. 영업 시간은 오전 10시 ~ 오후 6시입니다." 라고 대답하세요.

    context : {context}
    """),
    ("human", "{query}")
])


query = st.chat_input("질문을 입력해주세요.")

if query:
    chat_history(message=query, role="user")
    chain = {"context":retriever | RunnableLambda(combine_docs),
            "query":RunnablePassthrough()} | prompt | llm
    result = chain.invoke(query)
    chat_history(message=result.content, role = "ai")