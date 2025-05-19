import streamlit as st
import os
from dotenv import load_dotenv
# Assuming your_intelligence_module.py has the analyze_user_tone function
from your_intelligence_module import analyze_user_tone
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
from langchain.chains import LLMChain
# ============ Page Configuration ============
st.set_page_config(page_title="🎙️ AI SDR Voice Bot", layout="centered")

# ============ CSS Styling with Background Image ============
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1516321318423-f06f85e504b3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .main {
        max-width: 8px;
        margin: auto;
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        min-height: 0vh;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header img {
        width: 60px;
        vertical-align: middle;
        margin-right: 10px;
    }
    .bubble-user {
        background-color: #DCF8C6;
        padding: 12px 18px;
        border-radius: 16px;
        margin-bottom: 12px;
        max-width: 70%;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .bubble-user:hover {
        transform: translateY(-2px);
    }
    .bubble-bot {
        background-color: #E3F2FD;
        padding: 12px 18px;
        border-radius: 16px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 12px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .bubble-bot:hover {
        transform: translateY(-2px);
    }
    .spinner {
        text-align: center;
        font-size: 1.2rem;
        color: #0288D1;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============ Header ============
st.markdown("""
    <div class='header'>
        <img src='https://img.icons8.com/color/48/000000/microphone.png' alt='Mic Icon'/>
        <h1>🎧 AI SDR Voice Conversation</h1>
        <p style='color: #555;'>Engage with our AI Sales Development Representative in real-time!</p>
    </div>
""", unsafe_allow_html=True)

# Wrap the main app inside a centered container
st.markdown("<div class='main'>", unsafe_allow_html=True)

# ============ Environment Setup ============
load_dotenv()

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
        model_name="gemma2-9b-it",
        base_url="https://api.groq.com/openai/v1"
    )

    # Build retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    # Build RAG chain
    prompt = PromptTemplate.from_template(
         """
        You are an AI SDR (Sales Development Representative). Your goal is to qualify leads by asking short, clear, and helpful follow-up questions.

- Respond in exactly 1-2 sentences. Do not exceed 2 sentences.
- Keep your response concise, direct, and relevant to the user's query.
- Focus on qualifying the lead while maintaining a helpful tone.

Conversation History:
{history}

Context:
{context}

User just asked:
{question}

Your short response (strictly 1-2 sentences only):
        """
    )

    chain = RunnableMap({
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "history": RunnablePassthrough()
    }) | prompt | llm

    clean_prompt = PromptTemplate.from_template(
"""
You are a helpful AI assistant cleaning up voice-to-text input to make it clear, complete, and grammatically correct. 
Additionally, you are aware of the ongoing conversation and can use it to provide more contextually relevant and coherent responses.

- Consider the context from the conversation history to better understand the intent of the query
- Fix disfluencies (e.g., "um", "uh")
- Improve grammar
- If the sentence is incomplete or vague, complete it in a natural and helpful way, considering the context
- Normalize phrasing to standard question or request formats

Do not provide explanations or justifications for your changes. Only return the cleaned and contextual query.

Conversation History:
{history}

Current Voice Transcript:
"{transcript}"

Cleaned & Contextual Query:
"""
)

    clean_chain = RunnableMap({
        "history": RunnablePassthrough(),
        "transcript": RunnablePassthrough()
    }) | clean_prompt | llm
    return retriever, chain, clean_chain

retriever, response_chain, clean_chain = initialize_rag()

# ============ Audio Input & RAG Response ============
audio_file = record_audio()

if audio_file:
        user_text1 = transcribe_audio(audio_file)
        os.remove(audio_file)
        st.session_state.history.append((user_text1, "..."))
        formatted_history = "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in st.session_state.history]
        )
        user = clean_chain.invoke({
             "history": formatted_history,
             "transcript" : user_text1
        })
        user_text = user.content.strip()
        #print("user text:" , user_text)

        udocs = retriever.invoke(user_text)
        docs = udocs[:3]
        context = "\n".join([doc.page_content for doc in docs])

        response = response_chain.invoke({
            "context": context,
            "question": user_text,
            "history": formatted_history
        })
        #print("History: ",formatted_history)
        bot_text = response.content.strip()
        st.session_state.history[-1] = (user_text1, bot_text)

        bot_audio = generate_audio(bot_text)
        st.audio(bot_audio, format="audio/mpeg", autoplay=True)
            
        tone = analyze_user_tone(st.session_state.history)
        #st.info(f"🧠 Detected user tone: **{tone}**")
        if tone in ["frustrated", "interested"]:
            if len([u for u, _ in st.session_state.history if u != "__system__"]) > 3:
                st.toast("🚨 Escalating this lead to a human SDR.")
else:
        if not st.session_state.history:
            input = "Hello, how can I assist you today?"
            st.session_state.history.append(("__system__", input))
            bot_audio = generate_audio(input)
            st.audio(bot_audio, format="audio/mpeg", autoplay=False)

# ============ Display Chat History ============
for user_text, bot_text in st.session_state.history:
    if user_text == "__system__":
        st.markdown(f"""
        <div style='margin-bottom: 1.2rem;'>
            <div class="bubble-bot">
                <strong>🤖 Bot:</strong> {bot_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='margin-bottom: 1.2rem;'>
            <div class="bubble-user">
                <strong>🧑 You:</strong> {user_text}
            </div>
            <div class="bubble-bot">
                <strong>🤖 Bot:</strong> {bot_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
