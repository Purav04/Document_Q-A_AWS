import json, os
import streamlit as st
import boto3
import numpy as np

## importing embeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

## data ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## vector store
from langchain_community.vectorstores import FAISS

## llm model
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

## bedrock connection
bedrock = boto3.client(service_name="bedrock-runtime") 
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock)
current_directory = os.path.dirname(os.path.abspath(__file__))



## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader(f"{current_directory}/pdf files")
    documents = loader.load()

    ## text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.split_documents(documents)
    
    return docs

## Vector Embedding + Store
def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(embedding=bedrock_embeddings, documents=docs)
    vector_store_faiss.save_local(f"{current_directory}/faiss_embedding_store")

def get_mistral_llm():
    ## create mistral model
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",
                  client=bedrock,
                  model_kwargs={
                      "max_tokens":200,
                      "temperature":0.5,
                      "top_p":0.9,
                      "top_k":5
                  })
    
    return llm

def get_titan_llm():
    ## create mistral model
    llm = Bedrock(model_id="amazon.titan-text-lite-v1",
                  client=bedrock,
                  model_kwargs={
                      "maxTokenCount":4096,
                      "temperature":0.5,
                      "topP":0.9
                  })
    
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but ase atleast summarize 
with 250 words with detailed explanation. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def get_response_llm(query, llm, vector_store_faiss):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    answer = qa({"query": query})
    return answer["result"]

def main():
    st.set_page_config("CHAT PDF")
    st.header("Chat with PDF using AWS BEDROCK")

    response = ""
    user_question = st.text_input("Ask a Question from the PDF files")

    with st.sidebar:
        st.title("Create or Update Vector Store:")
        if st.button("Vector Create/Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(f"{current_directory}/faiss_embedding_store", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_mistral_llm()
            response = get_response_llm(user_question, llm, faiss_index)
            
            st.success("Done")
    
    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(f"{current_directory}/faiss_embedding_store", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_titan_llm()
            response = get_response_llm(user_question, llm, faiss_index)
            
            st.success("Done")
    
    
    st.write(response)

if __name__ == "__main__":
    main()

