import os
import wget
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import BSHTMLLoader
from langchain.chains import RetrievalQA

#download War and Peace by Tolstoy
wget.download("http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0073.shtml")

#load text from html
loader = BSHTMLLoader("text_0073.shtml", open_encoding='ISO-8859-1')
war_and_peace = loader.load()

#init Vector DB
embeddings = OpenAIEmbeddings()

doc_store = Qdrant.from_documents(
    war_and_peace, 
    embeddings,
    location=":memory:", 
    collection_name="docs",
)

llm = OpenAI()
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