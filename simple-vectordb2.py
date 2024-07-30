import os
import wget
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import BSHTMLLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader

# load the pdf and split it into chunks
loader = OnlinePDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
#load text from html
#loader = BSHTMLLoader("text_0073.shtml", open_encoding='ISO-8859-1')
#war_and_peace = loader.load()
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#all_splits = text_splitter.split_documents(war_and_peace)

doc_store = Chroma.from_documents(
    documents=all_splits, 
    embedding=HuggingFaceEmbeddings(),
    persist_directory='doc_store',
    collection_name="docs"
)
doc_store.persist()


llm = Ollama(model="mistral")
# ask questions

while True:
    question = input('Your question: ')
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_store.as_retriever(),
        return_source_documents=False,
    )

    result = qa(question)
    print(f"Answer: {result}")



