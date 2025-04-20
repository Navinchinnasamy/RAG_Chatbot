from modules.rag_model import RAGModel
from modules.context_builder import ContextBuilder
from modules.config_loader import load_config

class Chatbot:
    def __init__(self, config):
        self.rag_model = RAGModel(config)
        self.context_builder = ContextBuilder(config)

    def process_query(self, query):
        # Build context with conversation history
        context = self.context_builder.build_context(query)
        # Get response from RAG model
        response = self.rag_model.answer_query(context)
        # Log for debugging
        print(f"Query: {query}\nResponse: {response}")
        # Update context with new interaction
        self.context_builder.add_interaction(query, response)
        return response

    def reset_context(self):
        self.context_builder.clear_context()

if __name__ == "__main__":
    config = load_config()
    chatbot = Chatbot(config)
    queries = [
        "What is the interest rate of fixed deposit",
        "What are the documents required for gold loan"
    ]
    for query in queries:
        response = chatbot.process_query(query)
        print(f"Response: {response}\n")