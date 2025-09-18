import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

# API Keys
groq_api_key = st.secrets("GROQ_API_KEY")
os.environ["HF_TOKEN"] = st.secrets("HF_TOKEN")
#groq_api_key = os.getenv("GROQ_API_KEY")
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = st.secrets("LANGCHAIN_API_KEY")
#os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = st.secrets("LANGCHAIN_PROJECT")

# Embedding
embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")

# LLM
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model = "llama-3.1-8b-instant"  #"openai/gpt-oss-120b"
)

# Streamlit UI 

# Title
st.title("🗨️Q&a With Doc")
st.markdown("Upload any PDF and get your query sorted related to it's content!")

# Chat History for Stateful management
username = st.sidebar.text_input("Enter your username").replace(" ", "")
if not username:
     username = "default"
session_id = st.sidebar.text_input("Session ID", value= f"{username}_session")
if "store" not in st.session_state:
        st.session_state.store = {}

uploaded_files = st.sidebar.file_uploader("Upload Your PDF!", type= "pdf", accept_multiple_files=True)

# Chat Interface
if uploaded_files:
    
    documents = []
    for uploaded_file in uploaded_files:
        tempPdf = f"./temp.pdf"
        with open(tempPdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)

    # Splitting and create Embeddings for documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 5000, chunk_overlap= 500)
    splits = text_splitter.split_documents(documents= documents)
    vectorStore = Chroma.from_documents(documents= documents, embedding= embeddings)
    retirever = vectorStore.as_retriever()

    # Contextualized Prompt
    contextualize_que_system_prompt = (
        "You are a query reformulation assistant."
        "Your task is to process the most recent user question as follows:"
        "If the question depends on prior conversation, rewrite it into a complete, standalone query that contains all necessary context, entities, and details from the chat history."
        "If the question is already standalone and unambiguous, return it exactly as written."
        "Do not answer the question. Do not add explanations. Only return the reformulated query."
        "Your entire output must be the final reformulated query only."
    )

    contextualize_que_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_que_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retirever, contextualize_que_prompt)

    # QnA Prompt
    system_prompt = (
        """
        You are an AI assistant. Answer user questions using the retrieved document context as your primary source. Write naturally, as if speaking directly to the user.\n\n

        Rules:\n
        1. Use the retrieved context to answer directly in natural language.\n
        2. If the context is incomplete, you may supplement with general knowledge, but do not state this explicitly.\n
        3. If the information is not present, reply exactly: 'I could not find this information in the documents.'\n
        4. Keep answers concise, clear, and well-structured. Use bullet points or numbers for lists.\n
        5. Do not copy text verbatim from the context—always paraphrase and explain.\n
        6. Never mention phrases like 'based on the context' or 'according to the documents.'\n\n

        Retrieved Context:\n
        {context}
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # Create Chain
    qna_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

    # Session History fn
    def get_session_history(session:str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    # Create Conversational RAG Chain
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key= "input",
        history_messages_key= "chat_history",
        output_messages_key= "answer"
        )
    
    # User Input
    if user_input := st.chat_input("Ask your query:"):
        session_history = get_session_history(session_id)
        res = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            }
        )
        for msg in session_history.messages:
                with st.chat_message(msg.type):
                    st.markdown(msg.content)
        #st.write(st.session_state.store)
        #st.write("Naruto AI:", res["answer"])
        #st.write("chat_history:", session_history.messages)
else:
    st.info("👈 Please upload any PDF to continue!")


#st.caption("Made by Babuaa with 🎧")
