import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
#from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Setup ---
load_dotenv()
#sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Fix sqlite3 dependency for Chroma
os.environ["CHROMA_TELEMETRY"] = "False"

st.title("Auction.com Chatbot 🤖")
st.markdown(
    """
    This chatbot is trained on foreclosure documents provided by Mike's team.
    As we get more data, capabilities will improve significantly.
    """
)

# --- Caching heavy resources ---
@st.cache_resource
def get_vectorstore():
    loader = PyPDFLoader("Foreclosure_Full_Doc.pdf")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embedding=embeddings)
    #return Chroma.from_documents(docs, embedding=embeddings)

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# --- Initialize LLM + Prompt Chain (cached globally) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

system_prompt = (
    "You are an assistant answering questions posed by Auction.com employees. "
    "Use the retrieved context below to answer each question thoroughly. "
    "If you don't know the answer, say so clearly and request more data.\n\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- Chat interface ---
query = st.chat_input("Ask a question about foreclosure documents...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        result = rag_chain.invoke({"input": query})
        st.markdown(result["answer"])
