

import streamlit as st
import os
from speech_to_text import record_audio, transcribe_audio
from text_to_speech import generate_audio

# LangChain setup
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.retrievers.multi_query import MultiQueryRetriever

# ============ Streamlit Config ============
st.set_page_config(page_title="🎙️ AI SDR Voice Bot", layout="centered")

# ============ Page Title ============
st.markdown("""
    <h1 style='text-align: center;'>🎧 AI SDR Voice Conversation</h1>
""", unsafe_allow_html=True)

# ============ Chat History ============
if "history" not in st.session_state:
    st.session_state.history = []

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

    embedding = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    loader = TextLoader("company_faq.txt")
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.create_documents([full_text])
    vectordb = FAISS.from_documents(splits, embedding=embedding)

    llm = ChatOpenAI(
        openai_api_key="grok_api_key",
        temperature=0.0,
        model_name="mistral-saba-24b",
        base_url="https://api.groq.com/openai/v1"
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    prompt = PromptTemplate.from_template(
       "You are an AI SDR (Sales Development Representative). Your goal is to qualify leads by asking short, clear, and helpful follow-up questions."
       " Use the following conversation history and current context to respond. Keep your answer under 2 sentences.\n\n"
       "Conversation history:\n{history}\n\nContext:\n{context}\n\nUser just asked:\n{question}\n\nYour short response:"
    )

    chain = RunnableMap({
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough()
    }) | prompt | llm

    return retriever, chain

retriever, response_chain = initialize_rag()

# ============ Display Chat =============
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

# ============ Record on Button Press ============
st.markdown("""
    <style>
    .speak-button {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background-color: #25D366;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        z-index: 9999;
    }
    </style>
    <script>
    const doc = window.parent.document;
    const btn = doc.createElement("button");
    btn.innerHTML = "🎤 Speak";
    btn.className = "speak-button";
    btn.onclick = () => {
        const streamlitEvent = new CustomEvent("streamlit:sendMessage", {
            detail: {type: "speak_button_clicked"}
        });
        window.parent.dispatchEvent(streamlitEvent);
    };
    if (!doc.querySelector(".speak-button")) {
        doc.body.appendChild(btn);
    }
    </script>
""", unsafe_allow_html=True)

# Manual trigger via query param
if st.button("🎤 Speak"):
    audio_file = record_audio()
    st.audio(audio_file, format="audio/wav")

    user_text = transcribe_audio(audio_file)
    os.remove(audio_file)

    st.session_state.history.append((user_text, "..."))  # Placeholder

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