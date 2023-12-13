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


def index_from_storage(storage_dir, service_context):
    storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_dir),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_dir),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_dir),
    )

    # don't need to specify index_id if there's only one index in storage context
    index = load_index_from_storage(storage_context, service_context=service_context)

    return (index, storage_context)

if __name__ == '__main__':

    storage_dir = "./storage_context/wolfMQTT"

    # Create LLM, Context provider
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    (index, storage_context) = index_from_storage(storage_dir=storage_dir, service_context=service_context )

    # Create a query engine
    query_engine = index.as_query_engine()

    prompt = input('How can I help you ? > ')
    response = query_engine.query(prompt)

    print()
    print(response)
    print() 
    print('Bye👋')


    
