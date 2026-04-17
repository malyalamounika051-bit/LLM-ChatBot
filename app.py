import os
import gradio as gr
import traceback
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from Hugging Face Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini if key exists
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model safely
def get_llm():
    if not GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",   # stable model
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

llm = get_llm()

# Chat function
def predict(message, history):
    try:
        if not GOOGLE_API_KEY:
            return "❌ API key missing. Add it in Hugging Face Secrets."

        if not message.strip():
            return "⚠️ Please enter a message."

        # Direct model call (more stable than chain)
        response = llm.invoke(message)
        return response.content

    except Exception as e:
        error_msg = str(e)
        print("FULL ERROR:\n", traceback.format_exc())
        return f"❌ ERROR: {error_msg}"

# Gradio Chat UI
demo = gr.ChatInterface(
    fn=predict,
    title="✨ AI Chatbot (Gemini) ✨",
    description="Ask anything! Powered by Gemini + LangChain",
    examples=[
        "What is Artificial Intelligence?",
        "Explain Python in simple terms",
        "Write a short poem about nature"
    ]
)

# Launch
if __name__ == "__main__":
    demo.launch()
