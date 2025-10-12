# 📄 RAG Chatbot with Local LLM and Streamlit

> An end-to-end RAG (Retrieval-Augmented Generation) chatbot that answers questions about user-uploaded PDF documents. This project runs a powerful, open-source Large Language Model entirely on your local machine, ensuring privacy and eliminating the need for API keys.

## Features

- Interactive Chat Interface: Ask questions about your documents in a user-friendly web UI built with Streamlit.
- Local & Private: Powered by Google's `gemma-2b-it`, the entire AI pipeline runs locally. Your documents and queries never leave your machine.
- PDF Document Support: Upload any PDF file and the application will automatically process and "learn" its content.
- RAG Architecture: Utilizes the advanced RAG pattern to provide context-aware, accurate answers based on the document's content.

---

## Tech Stack

- Language: Python
- Core Frameworks: LangChain, Streamlit, Hugging Face Transformers
- Large Language Model (LLM): `google/gemma-2b-it`
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Database: FAISS (Facebook AI Similarity Search)
- Core Libraries: PyTorch, PyPDF, NumPy

---

## Setup and Installation

Follow these steps to set up the project locally.

1. Clone the Repository

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
```

````

_(Action Required:Replace `YourUsername` and `YourRepoName` with your actual GitHub details.)_

1. Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on Mac/Linux
# source venv/bin/activate
```

1. Install Dependencies

- All required packages are listed in the `requirements.txt` file.

<!-- end list -->

```bash
pip install -r requirements.txt
```

4. Log in to Hugging Face

- The Gemma model requires you to be authenticated with Hugging Face. First, accept the license terms on the [model page](https://huggingface.co/google/gemma-2b-it). Then, log in via the terminal.

<!-- end list -->

```bash
huggingface-cli login
# or the newer command:
# hf auth login
```

- Paste your Hugging Face access token when prompted.

---

\#\# Usage

1. Run the Streamlit App

- With your virtual environment active, run the following command:

<!-- end list -->

```bash
python -m streamlit run app.py
```

2. Use the Application

- Your web browser will open with the application running.
- Use the sidebar to upload a PDF file.
- Wait for the application to process the document and create the knowledge base.
- Ask questions about your document in the chat input box at the bottom.

---
````
