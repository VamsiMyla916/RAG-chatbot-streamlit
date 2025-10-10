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
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store

@st.cache_resource
def get_rag_chain():
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
        max_new_tokens=512, # Increased max tokens for more detailed answers
        temperature=0.2,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# --- Main App Logic ---

if uploaded_file is not None:
    if "vector_store" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner("Processing PDF..."):
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
            st.success("PDF processed and knowledge base created!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = get_rag_chain()
                
                # --- NEW: Professional Prompt Template ---
                # This uses the official chat template for TinyLlama for better results
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
                    retriever=st.session_state.vector_store.as_retriever(),
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=False # We don't need the source docs in the final output
                )
                
                result = qa_chain.invoke({"query": prompt})
                raw_response = result['result']
                
                # --- NEW: More Robust Cleaning Logic ---
                # This function will clean the text more reliably
                def clean_response(text):
                    # First, remove the "Helpful Answer:" prefix if it exists
                    if "Helpful Answer:" in text:
                        text = text.split("Helpful Answer:", 1)[1]
                    
                    # Then, remove any repeated context or instructions
                    context_marker = "Use the following pieces of context"
                    if context_marker in text:
                        text = text.split(context_marker, 1)[0]
                        
                    return text.strip()

                cleaned_response = clean_response(raw_response)

                formatted_response = f"**Question:** {prompt}\n\n**Answer:**\n{cleaned_response}"
                
                st.markdown(formatted_response)
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
else:
    st.info("Please upload a PDF file to begin chatting.")

