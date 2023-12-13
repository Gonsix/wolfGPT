import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor
from llama_index import VectorStoreIndex, PromptHelper, ServiceContext, StorageContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index import load_index_from_storage
from pathlib import Path
from llama_index import download_loader
from langchain.chat_models import ChatOpenAI
from index_from_storage import index_from_storage
from create_index import create_index

st.title("Ask about wolfMQTT")

WOLFMQTT_DOC_PATH = "./docs/ja/wolfMQTT-Manual-jp.pdf"
STORAGE_DIR = './storage_context/wolfMQTT'


# Create LLM, Context provider
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Loed Index from Storage
(index, storage_context) = index_from_storage(STORAGE_DIR, service_context=service_context)

# Create a query engine
# Streaming = True にしてもストリーミングできない
chat_engine = index.as_chat_engine(streaming=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = chat_engine.chat(prompt)
            st.markdown(response_stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream})

