# 🗨️ Q&A With Doc

A **Streamlit-based conversational app** that lets you **upload PDFs** and interact with them in natural language.  
The system uses **LangChain**, **FAISS vector store**, **HuggingFace embeddings**, and **Groq LLMs** to retrieve and answer your queries from uploaded documents.  
It also supports **chat history**, enabling context-aware conversations.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing** – Upload one or more PDFs and extract content.
- ✂️ **Smart Text Splitting** – Splits large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- 🔎 **Context-Aware Retrieval** – Retrieves relevant passages using FAISS embeddings.
- 🤖 **Conversational Q&A** – Chat with the documents using `Groq LLM` (Llama 3.1-8B-Instant).
- 🧠 **History-Aware Queries** – Reformulates queries using chat history for better contextual answers.
- 🔄 **Session Management** – Multi-user and session-based conversations stored in Streamlit state.
- 🎯 **Concise & Natural Responses** – Answers are paraphrased, structured, and user-friendly.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Interactive web app UI
- [LangChain](https://www.langchain.com/) – Chains, retrievers, prompts
- [FAISS](https://faiss.ai/) – Vector database for similarity search
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) – Document embeddings
- [Groq LLM](https://groq.com/) – Fast & efficient language model inference
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) – PDF document parsing
- [Dotenv](https://pypi.org/project/python-dotenv/) – Environment variable management

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/qna-with-doc.git
cd qna-with-doc
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
```

If deploying on **Streamlit Cloud**, add these keys to `st.secrets`.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📌 Usage

1. Open the app in your browser after running.
2. Enter a **username** and **session ID** from the sidebar.
3. Upload one or more **PDF files**.
4. Ask any question in the chat box.
5. The assistant will retrieve relevant information and provide clear answers.

---

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (ignored in Git)
└── README.md           # Documentation
```
