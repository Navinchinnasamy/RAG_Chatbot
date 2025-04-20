import streamlit as st
from modules.chatbot import Chatbot
from modules.config_loader import load_config

# Initialize Streamlit app
st.title("Website Chatbot")
st.write("Ask questions about the website content!")

# Load configuration and initialize chatbot
config = load_config()
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot(config)
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input("Your question:", "")
    submit_button = st.form_submit_button("Send")

# Process query
if submit_button and user_query:
    response = st.session_state.chatbot.process_query(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "bot", "content": response})

# Display conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Bot**: {message['content']}")

# Reset conversation
if st.button("Reset Conversation"):
    st.session_state.chatbot.reset_context()
    st.session_state.messages = []
    st.write("Conversation reset.")

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Ensure all modules (web_scraper, knowledgebase_builder, embedding_generator, vector_store) have been run.")
st.sidebar.write("2. Enter your question and click 'Send'.")
st.sidebar.write("3. Click 'Reset Conversation' to start a new session.")