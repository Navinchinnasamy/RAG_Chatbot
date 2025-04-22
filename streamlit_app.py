import streamlit as st
from modules.rag_model import RAGModel
from modules.config_loader import load_config

st.title("Shriram Finance Chatbot")
st.write("Ask about Shriram Finance products and services")

# Load configuration and initialize RAG model
config = load_config()
rag = RAGModel(config)

# Initialize session state for query history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# User input
query = st.text_input("Enter your question:", key="query_input")

if query:
    # Append query to history
    st.session_state.query_history.append(query)
    st.write(f"**Debug: Query History:** {st.session_state.query_history[-3:]}")  # Show last 3 queries
    
    # Get response
    response = rag.answer_query(query)
    st.write(f"**Question:** {query}")
    st.write(f"**Answer:** {response}")

# Display query history
st.subheader("Recent Queries")
for q in st.session_state.query_history[-5:]:
    st.write(f"- {q}")