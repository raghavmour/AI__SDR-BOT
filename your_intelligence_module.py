from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import streamlit as st
def analyze_user_tone(history):
    from langchain_openai import ChatOpenAI

    # Reuse your Mistral LLM on Groq
    llm = ChatOpenAI(
        openai_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.0,
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        base_url="https://api.groq.com/openai/v1"
    )

    # Format conversation history
    

    # Prompt for one-word tone classification
    prompt = PromptTemplate.from_template("""
Analyze the following user and bot conversation history. 

Conversation History:
{history}

Classify the user's tone using **only one word**, choosing from: "interested", "frustrated", "confused", or "neutral".

Respond with just the one word.
""")

    # Run the chain
    chain = RunnableMap({"history": RunnablePassthrough()}) | prompt | llm

    # Get response from model
    result = chain.invoke(history)

    tone = result.content.strip().lower()
    if tone not in ["interested", "frustrated", "confused", "neutral"]:
        tone = "neutral"  # Fallback in case of unexpected output

    return tone
