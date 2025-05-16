import streamlit as st
import os
from dotenv import load_dotenv

# ============ Page Configuration ============
st.set_page_config(page_title="🎙️ AI SDR Voice Bot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎧 AI SDR Voice Conversation</h1>", unsafe_allow_html=True)

# ============ Environment Setup ============
load_dotenv()

# ============ Imports ============
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.retrievers.multi_query import MultiQueryRetriever

from speech_to_text import record_audio, transcribe_audio
from text_to_speech import generate_audio

# ============ Chat History ============
if "history" not in st.session_state:
    st.session_state.history = []
import openai

# Set the API key directly


# ============ RAG Initialization ============
@st.cache_resource
def initialize_rag():
    class SentenceTransformerEmbeddings:
        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts):
            return self.model.encode(texts, convert_to_numpy=True).tolist()

        def embed_query(self, text):
            return self.model.encode(text, convert_to_numpy=True).tolist()

        def __call__(self, text):
            return self.embed_query(text)

    # Load and embed documents
    embedding = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    loader = TextLoader("company_faq.txt")
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.create_documents([full_text])
    vectordb = FAISS.from_documents(splits, embedding=embedding)

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=st.secrets["GROQ_API_KEY"],  # Load from .env
        temperature=0.0,
        model_name="mistral-saba-24b",
        base_url="https://api.groq.com/openai/v1"
    )

    # Build retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    # Build RAG chain
    prompt = PromptTemplate.from_template(
        "You are an AI SDR (Sales Development Representative). Your goal is to qualify leads by asking short, clear, and helpful follow-up questions."
        "\nUse the following conversation history and current context to respond. Answer in 2-4 sentences.\n\n"
        "Conversation history:\n{history}\n\nContext:\n{context}\n\nUser just asked:\n{question}\n\nYour short response:"
    )

    chain = RunnableMap({
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough()
    }) | prompt | llm

    return retriever, chain

retriever, response_chain = initialize_rag()

# ============ Display Chat History ============


# ============ Audio Input & RAG Response ============
audio_file = record_audio()

if audio_file:
    # st.audio(audio_file, format="audio/wav")  # Optional playback

    user_text = transcribe_audio(audio_file)
    os.remove(audio_file)

    st.session_state.history.append((user_text, "..."))  # Placeholder for bot response

    formatted_history = "\n".join(
        [f"User: {q}\nBot: {a}" for q, a in st.session_state.history[-3:-1]]
    )

    docs = retriever.invoke(user_text)
    context = "\n".join([doc.page_content for doc in docs])

    response = response_chain.invoke({
        "context": context,
        "question": user_text,
        "history": formatted_history
    })

    bot_text = response.content.strip()
    st.session_state.history[-1] = (user_text, bot_text)

    bot_audio = generate_audio(bot_text)
    st.audio(bot_audio, format="audio/mpeg", autoplay=True)
    for user_text, bot_text in st.session_state.history:
        st.markdown(f"""
        <div style='margin-bottom: 1rem;'>
            <div style='background-color: #DCF8C6; padding: 10px 15px; border-radius: 12px; margin-bottom: 4px; max-width: 75%;'>
                <strong>You:</strong> {user_text}
            </div>
            <div style='background-color: #F1F0F0; padding: 10px 15px; border-radius: 12px; margin-left: auto; max-width: 75%;'>
                <strong>Bot:</strong> {bot_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No audio was recorded. Please try again.")
