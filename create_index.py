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

def create_index(docfile_path, storage_dir, service_context):
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(docfile_path))
    # create an index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Save an index to storage ## If this program was executed by python <this program> new
    storage_context = index.storage_context
    storage_context.persist(persist_dir=storage_dir, vector_store_fname="vector_store.json")
    
    return (index, storage_context)


if __name__ == '__main__':
    # Create LLM, Context provider
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docfile_path = "./docs/ja/wolfBoot-Manual-jp.pdf"
    storage_dir = "./storage_context/wolfBoot"
    (index, storage_context) = create_index(docfile_path, storage_dir, service_context)


