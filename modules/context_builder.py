from collections import deque
from modules.config_loader import load_config

class ContextBuilder:
    def __init__(self, config):
        self.context_window = config["chatbot"]["context_window"]
        self.history = deque(maxlen=self.context_window)

    def add_interaction(self, query, response):
        self.history.append({"query": query, "response": response})

    def build_context(self, current_query):
        context = ""
        for interaction in self.history:
            context += f"User: {interaction['query']}\nBot: {interaction['response']}\n"
        context += f"User: {current_query}\n"
        return context

    def clear_context(self):
        self.history.clear()

if __name__ == "__main__":
    config = load_config()
    context_builder = ContextBuilder(config)
    context_builder.add_interaction("What is the website about?", "It's about tech news.")
    context = context_builder.build_context("Tell me more.")
    print(context)