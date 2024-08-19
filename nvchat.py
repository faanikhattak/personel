import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
import warnings
import base64
import time

warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`.")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`.")

# AI Voice Assistant Class
class SimpleRAGAssistant:
    def __init__(self, file_path):
        try:
            # Initialize Ollama with the correct parameter
            self._llm = Ollama(model="mistral", request_timeout=120.0)

            # Initialize HuggingFace embedding model (example)
            self._embedding_model = HuggingFaceEmbedding(model_name="bert-base-uncased")  # Example model name

            # Initialize ServiceContext with default settings
            self._service_context = ServiceContext.from_defaults(
                llm=self._llm,
                embed_model=self._embedding_model,
            )
            
            # Initialize the knowledge base and chat engine
            self._index = None
            self._create_kb(file_path)  # Ensure knowledge base is created before chat engine
            self._create_chat_engine()
        except Exception as e:
            st.error(f"Error during initialization: {e}")

    def _create_chat_engine(self):
        if self._index is None:
            st.error("Index has not been initialized. Ensure _create_kb() is called successfully.")
            return
        
        try:
            # Initialize chat engine
            self._chat_engine = self._index.as_chat_engine(
                chat_mode="context",
                system_prompt=self._prompt,
            )
        except Exception as e:
            st.error(f"Error while creating chat engine: {e}")

    def _create_kb(self, file_path):
        if not os.path.isfile(file_path):
            st.error(f"File {file_path} does not exist.")
            return

        try:
            reader = SimpleDirectoryReader(
                input_files=[file_path]  # Use the correct path
            )
            documents = reader.load_data()
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context
            )
            st.success("Knowledgebase created successfully!")
        except Exception as e:
            st.error(f"Error while creating knowledgebase: {e}")
            self._index = None  # Ensure _index is set to None in case of failure

    def interact_with_llm(self, customer_query):
        # Ensure _chat_engine is properly initialized
        if not hasattr(self, '_chat_engine') or self._chat_engine is None:
            st.error("Chat engine not initialized.")
            return "Chat engine not initialized."
        try:
            AgentChatResponse = self._chat_engine.chat(customer_query)
            return AgentChatResponse.response
        except Exception as e:
            st.error(f"Error during interaction with LLM: {e}")
            return "An error occurred during the interaction."

    @property
    def _prompt(self):
        return """
            You are a professional AI Assistant. Answer questions based on the provided knowledge base.
            If you don't know the answer, just say that you don't know. Keep responses concise and to the point.
            """

# Function to convert text to speech and play it
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode()
        return audio_base64
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

# Streamlit App
def main():
    st.markdown("<h1 style='color: #FF6347; text-align: center;'>USER WITH AI VOICE ASSISTANT APP</h1>", unsafe_allow_html=True)
    st.write("Enter your query below or use voice input:")

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    
    if uploaded_file is not None:
        # Ensure the directory exists
        temp_dir = "tempDir"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize the AI Assistant
        ai_assistant = SimpleRAGAssistant(file_path)

        # Text input for user query
        user_input = st.text_input("Your Query:")

        if st.button("Get Response"):
            if user_input:
                output = ai_assistant.interact_with_llm(user_input)
                st.write(f"AI Assistant: {output}")
                audio_base64 = text_to_speech(output)
                if audio_base64:
                    time.sleep(2)  # Delay for 2 seconds
                    st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")

        # Voice input functionality
        st.write("Or, use your voice to ask a question:")

        recognizer = sr.Recognizer()

        if st.button("Record Voice"):
            try:
                with sr.Microphone() as source:
                    st.write("Listening...")
                    audio = recognizer.listen(source)
                    st.write("Processing...")
                    user_input = recognizer.recognize_google(audio)
                    st.write(f"You said: {user_input}")
                    output = ai_assistant.interact_with_llm(user_input)
                    st.write(f"AI Assistant: {output}")
                    audio_base64 = text_to_speech(output)
                    if audio_base64:
                        time.sleep(2)  # Delay for 2 seconds
                        st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
            except sr.RequestError:
                st.error("Sorry, there was an issue with the speech recognition service.")
    
    # Footer
    st.markdown(
        "<footer style='text-align: center; color: white; background-color: #4CAF50; padding: 10px;'>"
        "Developed by Irfan Khattak</footer>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
