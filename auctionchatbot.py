import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

st.title("Auction.com Chatbot")
st.write("This chatbot is trained on 2 foreclosure documents provided by Mike's Team. As we get more training data, this chatbots capabilities can be significantly enhanced") 
st.write("Answers will be displayed below this line")

loader = PyPDFLoader("Foreclosure_Full_Doc.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)


vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=1,max_tokens=None,timeout=None)


query = st.chat_input("Hello Auction.com Team! Go ahead and ask something") 
prompt = query

system_prompt = (
    "You are an assistant for answering questions posed by auction.com employees. Ensure that all answers are related to Auction.com data thats provided to you "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know and need more data. There is no restriction on number of words to answer any question and you can be as detailed and thorough as possible"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    #print(response["answer"])
    st.write("User Asked: "+query)
    st.write("Chatbot Response: "+response["answer"])
