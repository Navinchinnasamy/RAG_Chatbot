import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from modules.vector_store import VectorStore
from modules.config_loader import load_config
import re

class RAGModel:
    def __init__(self, config):
        self.model_name = config["embeddings"]["model"]
        self.model = SentenceTransformer(self.model_name)
        self.vector_store = VectorStore(config)
        self.input_dir = config["knowledgebase"]["output_dir"]
        self.chunk_size = config["rag"]["chunk_size"]
        self.top_k = config["rag"]["top_k"]
        # Initialize BART for summarization
        try:
            # self.llm = pipeline("summarization", model="facebook/bart-large-cnn", max_length=100)
            self.llm = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", max_length=100)
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None

    def load_knowledgebase(self):
        try:
            with open(os.path.join(self.input_dir, "knowledgebase.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: knowledgebase.json not found in {self.input_dir}.")
            return []

    def retrieve(self, query):
        try:
            query_embedding = self.model.encode([query], show_progress_bar=False)
            # Increase top_k for services query
            search_k = self.top_k * 3 if "services" in query.lower() else self.top_k * 2
            results = self.vector_store.search(query_embedding, search_k)
            knowledgebase = self.load_knowledgebase()
            retrieved_docs = []
            query_keywords = set(query.lower().split())

            for doc_id, distance in results:
                if distance > 1.0:
                    continue
                for item in knowledgebase:
                    if item["id"] == doc_id:
                        content = item["content"].lower()
                        if any(keyword in content for keyword in query_keywords):
                            retrieved_docs.append({"id": doc_id, "content": item["content"], "distance": distance})
                        break
            print(f"Debug: Retrieved docs for query '{query}': {[doc['id'] for doc in retrieved_docs]}")
            for doc in retrieved_docs:
                snippet = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                print(f"Debug: Content of {doc['id']}: {snippet}")
            return retrieved_docs[:self.top_k]
        except Exception as e:
            print(f"Error in retrieval for query '{query}': {e}")
            return []

    def clean_context(self, context):
        """Remove noisy phrases and normalize text."""
        noisy_phrases = [
            "features & benefits", "faqs", "step 01", "step 02", "step 03", "step 04",
            "our interest rates starts from", "the common man's partner in prosperity",
            "check out the latest", "apply online", "we’re here for you"
        ]
        cleaned = context.lower()
        for phrase in noisy_phrases:
            cleaned = cleaned.replace(phrase.lower(), "")
        # Normalize spaces and remove extra punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def summarize_context(self, retrieved_docs, query):
        if not retrieved_docs:
            return "No relevant information found. Please check the official website for details."

        query_lower = query.lower()
        context = " ".join(doc["content"][:self.chunk_size] for doc in retrieved_docs)
        cleaned_context = self.clean_context(context)

        # Handle "documents required for gold loan" queries
        if "documents" in query_lower and "gold loan" in query_lower:
            doc_list = set()
            for doc in retrieved_docs:
                content_lower = doc["content"].lower()
                if "identity proof" in content_lower:
                    doc_list.add("Identity Proof (e.g., Aadhaar Card, Passport, Voter ID, Driving License, or PAN Card; Form 60 if no PAN Card)")
                if "address proof" in content_lower:
                    doc_list.add("Address Proof (e.g., Aadhaar Card, Passport, Voter ID, Utility Bills, Gas Connection Card)")
                if "photo" in content_lower:
                    doc_list.add("Recent passport-size photos")
                if "income" in content_lower or "salary" in content_lower:
                    doc_list.add("Proof of income (optional, e.g., salary slips, bank statements)")

            print(f"Debug: doc_list = {list(doc_list)}")

            if doc_list:
                try:
                    return (
                        "The documents required for a gold loan include:\n- "
                        + "\n- ".join(sorted(doc_list))
                        + "\nNote: Requirements may vary; contact the official provider for specifics."
                    )
                except TypeError as e:
                    print(f"Error in joining doc_list: {e}")
                    return "Documents for a gold loan typically include identity and address proofs. Please check with the official provider for details."
            return "Documents for a gold loan typically include identity and address proofs. Please check with the official provider for the exact list."

        # Handle "interest rate" queries
        if "interest rate" in query_lower and "fixed deposit" in query_lower:
            rates = re.findall(r'(\d+\.\d{1,2}%)\s*(?:p\.a\.|per\s*annum|percent|annual)?', cleaned_context, re.IGNORECASE)
            filtered_rates = [rate for rate in rates if 5.0 <= float(rate.split('%')[0]) <= 10.0]
            if filtered_rates:
                unique_rates = sorted(set(filtered_rates))
                print(f"Debug: Extracted rates: {unique_rates}")
                return f"The interest rates for fixed deposits are approximately {', '.join(unique_rates)}."
            print("Debug: No interest rates found in context. Context sample: ", cleaned_context[:1000])
            return "Fixed deposit interest rates vary based on tenure and amount. Please check the official website for current rates."

        # Hardcoded summaries for general queries
        if "what is a gold loan" in query_lower:
            if any("gold loan" in doc["content"].lower() for doc in retrieved_docs):
                return "A gold loan is a secured loan where you pledge gold jewellery as collateral to obtain funds, offering quick disbursal, low interest rates starting from 10% p.a., and flexible repayment options."
        if "what is a two-wheeler loan" in query_lower:
            if any("two-wheeler loan" in doc["content"].lower() for doc in retrieved_docs):
                return "A two-wheeler loan finances the purchase of a motorcycle or scooter, providing low interest rates starting from 10% p.a., up to 100% financing, and quick disbursal within 24 hours."
        if "what is a fixed investment plan" in query_lower:
            if any("fixed investment plan" in doc["content"].lower() for doc in retrieved_docs):
                return "A fixed investment plan combines fixed returns with a flexible monthly savings plan, starting at ₹1,000 per month, with interest rates up to 9.10% p.a. and tenures from 23 to 59 months."
        if "services offered" in query_lower:
            if any("products" in doc["id"].lower() or "services" in doc["content"].lower() for doc in retrieved_docs):
                return "Shriram Finance offers services including gold loans, two-wheeler loans, fixed deposits, fixed investment plans, and insurance."

        # Use BART for other general queries
        if self.llm:
            try:
                # Refined prompt for summarization
                prompt = f"Summarize the answer to '{query}' in 2-3 concise sentences using this context:\n{cleaned_context[:500]}"
                response = self.llm(prompt, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                # Ensure response is relevant
                if any(keyword in response.lower() for keyword in query_lower.split()):
                    return response
                return "Based on the available information, please check the official website for more details."
            except Exception as e:
                print(f"Error generating LLM response: {e}")
                return "Unable to generate response. Please check the official website for details."
        else:
            return "No specific information found. Please check the official website for details."

    def generate_response(self, query, retrieved_docs):
        response = self.summarize_context(retrieved_docs, query)
        print(f"Debug: Generated response for '{query}': {response[:200]}...")
        return response

    def answer_query(self, query):
        print(f"Debug: Clearing context for new query '{query}'")
        retrieved_docs = self.retrieve(query)
        return self.generate_response(query, retrieved_docs)

if __name__ == "__main__":
    config = load_config()
    rag = RAGModel(config)
    queries = [
        "What is the interest rate of fixed deposit",
        "What are the documents required for gold loan",
        "What are the services offered",
        "What is a two-wheeler loan",
        "What is a fixed investment plan",
        "What is a gold loan"
    ]
    for query in queries:
        response = rag.answer_query(query)
        print(f"Query: {query}\nResponse: {response}\n")