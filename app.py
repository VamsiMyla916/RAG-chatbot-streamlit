import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="📄 Chat with Your Documents",
    page_icon="📄",
    layout="wide"
)

# --- App Title ---
st.title("📄 Chat with Your Documents")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# --- Functions ---

# Function to create the vector store
def get_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store

# Function to load the LLM, prepared for deployment
@st.cache_resource
def get_rag_chain():
    model_id = "distilgpt2"
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get the Hugging Face token from Streamlit secrets
    # This is the secure way to handle secrets in a deployed app
    hf_token = st.secrets["HUGGING_FACE_HUB_TOKEN"]

    # Load the tokenizer and model using the token
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto", 
        trust_remote_code=True,
        token=hf_token
    ).to(device)

    # Create the text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
    )
    
    # Wrap the pipeline in a LangChain object
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

# --- Main App Logic ---

# 1. Process the Uploaded File
if uploaded_file is not None:
    # This check ensures the PDF is processed only once per upload
    if "vector_store" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        with st.spinner("Processing PDF..."):
            # Create a temporary directory to store the file
            temp_dir = "temp_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and chunk the document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # Create and save the vector store in the session state
            st.session_state.vector_store = get_vector_store(chunks)
            st.success("PDF processed and knowledge base created!")

    # 2. Initialize and Display Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle User Input and Generate Response
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Load the cached LLM
                llm = get_rag_chain()
                
                # Create the RAG chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )
                
                # Get the answer
                response = qa_chain.run(prompt)
                st.markdown(response)
        
        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a PDF file to begin chatting.")

