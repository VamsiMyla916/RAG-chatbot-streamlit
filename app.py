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
        Made with ❤️ by **Vamsi Krishna Sai Myla**
        
        **Connect & Provide Feedback:**
        - [LinkedIn](https://www.linkedin.com/in/vamsimyla/)
        - [GitHub](https://github.com/VamsiMyla916/RAG-chatbot-streamlit)
        - [Email](mailto:mylavamsikrishnasai@gmail.com)
        """
    )

# --- Functions ---

def get_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store

@st.cache_resource
def get_rag_chain():
    # --- MODEL FOR DEPLOYMENT ---
    # Using a very small model to fit into Streamlit's free hardware
    model_id = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # .get() is used for graceful error handling if the secret is missing
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
        max_new_tokens=256, # Reduced max tokens for the smaller model
        temperature=0.7,
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
                
                # A simpler prompt template suitable for the smaller distilgpt2 model
                prompt_template = """
                Use the following context to answer the question. If you don't know the answer, say you don't know.
                Context: {context}
                Question: {question}
                Answer:"""
                
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever(),
                    chain_type_kwargs={"prompt": PROMPT},
                )
                
                result = qa_chain.invoke({"query": prompt})
                response = result.get('result', "I couldn't find an answer.").strip()

                # Clean the response to prevent the model from repeating the prompt
                if "Question:" in response:
                    response = response.split("Question:")[0].strip()
                if "Answer:" in response:
                    response = response.split("Answer:")[1].strip()

                formatted_response = f"**Question:** {prompt}\n\n**Answer:**\n{response}"
                
                st.markdown(formatted_response)
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
else:
    st.info("Please upload a PDF file to begin chatting.")

