import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from google import genai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


from dotenv import load_dotenv
load_dotenv()

import os
api_key=os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )
    st.divider()
    st.markdown("### 📘 Assistant Capabilities")
    st.markdown("""
- 📄 Understands and processes PDF documents
- 🔍 Finds relevant content using smart semantic search
- 🤖 Answers questions with context-aware AI
- ⚡ Uses Gemini Models for fast and accurate responses
""")
    st.divider()
    st.markdown("Built with [Gemini](https://ai.google.dev/gemini-api/docs/models) | [LangChain](https://python.langchain.com/)")


PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH="document Store/pdfs/"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 

class SBERTEmbeddings:
    def embed_documents(self, texts):
        return EMBEDDING_MODEL.encode(texts).tolist() 
    
    def embed_query(self, text):
        return EMBEDDING_MODEL.encode([text])[0].tolist()

DOCUMENT_VECTOR_DB = InMemoryVectorStore(SBERTEmbeddings())


def save_uploaded_file(uploaded_file):
    file_path=PDF_STORAGE_PATH+uploaded_file.name
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_document(file_path):
    document_loader=PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents, selected_model=selected_model):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, document_context=context_text)
    model = genai.GenerativeModel(selected_model)
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else str(response)




# UI Configuration

st.title("📘 MindPaper AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section

uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_document(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
