import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

# --- Sidebar for File Upload & Contact Info ---
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.divider()
    st.markdown(
        """
        **Please provide your valuable feedback or suggestions on how can we further improve this application. We can connect and discuss. Please find my linkedin and Github here:**
        - [LinkedIn](https://www.linkedin.com/in/vamsimyla/)
        - [GitHub](https://github.com/VamsiMyla916/RAG-chatbot-streamlit)
        - [Email:mylavamsikrishnasai@gmail.com](mailto:mylavamsikrishnasai@gmail.com)
        """
    )

# --- Functions ---

def get_vector_store(chunks):
    """Creates a FAISS vector store from document chunks."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store

@st.cache_resource
def get_llm_pipeline():
    """Initializes and returns the language model pipeline."""
    model_id = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = st.secrets.get("HUGGING_FACE_HUB_TOKEN", "")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token
    ).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# --- Main App Logic ---

if uploaded_file is not None:
    # Process the PDF and create the vector store if it's a new file
    if "vector_store" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner("Processing PDF... This may take a moment."):
            temp_dir = "temp_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            st.session_state.vector_store = get_vector_store(chunks)
            st.success("✅ PDF processed and knowledge base created!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Display user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                llm = get_llm_pipeline()
                
                prompt_template = """
                <|system|>
                You are a helpful AI assistant. Use the provided context to answer the user's question.
                If the answer is a list, format it with markdown bullets.
                If you don't know the answer, simply state that you don't know.
                Do not repeat the question or the context in your answer. Provide only the helpful answer itself.
                Context: {context}</s>
                <|user|>
                Question: {question}</s>
                <|assistant|>
                Helpful Answer:"""
                
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    # FIX: Limit retriever to top 2 results to avoid context overflow error
                    retriever=st.session_state.vector_store.as_retriever(search_kwargs={'k': 2}),
                    chain_type_kwargs={"prompt": PROMPT},
                )
                
                result = qa_chain.invoke({"query": prompt})
                raw_response = result['result']
                
                # --- NEW: More Robust Cleaning Logic ---
                def clean_response(text):
                    # Remove the "Helpful Answer:" prefix and any leading/trailing whitespace
                    if "Helpful Answer:" in text:
                        text = text.split("Helpful Answer:", 1)[1]
                    
                    # Remove any text that looks like a repeated instruction or context
                    stop_phrases = ["<|system|>", "<|user|>", "Context:", "Question:"]
                    for phrase in stop_phrases:
                        if phrase in text:
                            text = text.split(phrase, 1)[0]
                            
                    return text.strip()

                cleaned_response = clean_response(raw_response)

                # --- MODIFIED: Display only the clean answer ---
                st.markdown(cleaned_response)
        
        # --- MODIFIED: Save only the clean answer to history ---
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
else:
    st.info("👋 Welcome! Please upload a PDF file to begin chatting.")