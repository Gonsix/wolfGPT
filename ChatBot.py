import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.schema import HumanMessage, AIMessage, SystemMessage,  StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.callbacks import StreamlitCallbackHandler

from streamlit_callback import StreamHandler
from retriever import get_retriever
from router import get_router
from logger import get_logger


logger = get_logger()

WOLFSSL_ICON_PATH = './image/wolfssl-icon.png'


RAG_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""



prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
retriever = get_retriever()
router = get_router()
llm = ChatOpenAI(streaming=True, model='gpt-4-turbo-preview')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Chat wolfSSL")
st.caption("🚀 Document Retrieval chatbot powered by OpenAI GPT-4 turbo")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    if msg.role == "assistant":
        st.chat_message('assistant', avatar=WOLFSSL_ICON_PATH).write(msg.content)
    else:
        st.chat_message('user').write(msg.content)

if usr_query := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=usr_query))
    st.chat_message("user").write(usr_query)


    with st.chat_message("assistant", avatar=WOLFSSL_ICON_PATH):
        # logger.debug(f'route: {router(prompt)}')    
        query_type = router(usr_query).name
        # print('route: ', router(prompt).name)    
        logger.info(f"Router: decided to route {query_type}")
        if query_type is None:

            response = "Hmm, I'm not sure 🤐"
            st.write(response)

        else:
            stream_handler = StreamHandler(st.empty()) # Stream handler should be created everytime before chian.invoke()
            response = rag_chain.invoke(usr_query, {"callbacks": [stream_handler]})

        # print("response: ", response)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))
