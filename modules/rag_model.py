import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
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
            results = self.vector_store.search(query_embedding, self.top_k * 2)
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

    def summarize_context(self, retrieved_docs, query):
        if not retrieved_docs:
            return "No relevant information found. Please check the official Shriram Finance website for details."

        query_lower = query.lower()
        context = " ".join(doc["content"][:self.chunk_size] for doc in retrieved_docs)

        # Handle "interest rate" queries
        if "interest rate" in query_lower and "fixed deposit" in query_lower:
            rates = re.findall(r'(\d+\.?\d*%\s*(?:p\.a\.|per\s*annum|percent|annual)?)', context, re.IGNORECASE)
            if rates:
                unique_rates = sorted(set(rates))
                print(f"Debug: Extracted rates: {unique_rates}")
                return f"The interest rates for fixed deposits at Shriram Finance are approximately {', '.join(unique_rates)}."
            print("Debug: No interest rates found in context. Context sample: ", context[:1000])
            return "Fixed deposit interest rates at Shriram Finance vary based on tenure and amount. Please check the official website for current rates."

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
                        "The documents required for a gold loan at Shriram Finance include:\n- "
                        + "\n- ".join(sorted(doc_list))
                        + "\nNote: Requirements may vary; contact Shriram Finance for specifics."
                    )
                except TypeError as e:
                    print(f"Error in joining doc_list: {e}")
                    return "Documents for a gold loan typically include identity and address proofs. Please check with Shriram Finance for details."
            return "Documents for a gold loan typically include identity and address proofs. Please check with Shriram Finance for the exact list."

        # Handle "services offered" queries
        if "services" in query_lower and "shriram" in query_lower:
            services = set()
            for doc in retrieved_docs:
                content_lower = doc["content"].lower()
                if "fixed deposit" in content_lower:
                    services.add("Fixed Deposits")
                if "gold loan" in content_lower:
                    services.add("Gold Loans")
                if "two-wheeler" in content_lower or "two wheeler" in content_lower:
                    services.add("Two-Wheeler Loans")
                if "insurance" in content_lower:
                    services.add("Insurance")
                if "invest" in content_lower:
                    services.add("Investment Plans")
            if services:
                return (
                    "Shriram Finance offers the following services:\n- "
                    + "\n- ".join(sorted(services))
                    + "\nFor more details, visit the official Shriram Finance website."
                )
            return "Shriram Finance offers various financial services including loans and investments. Please check the official website for a complete list."

        # Handle "two-wheeler loan" queries
        if "two-wheeler" in query_lower or "two wheeler" in query_lower:
            for doc in retrieved_docs:
                content_lower = doc["content"].lower()
                if "two-wheeler" in content_lower or "two wheeler" in content_lower:
                    return (
                        "A Two-Wheeler Loan from Shriram Finance is a financial product designed to help you purchase a motorcycle or scooter. "
                        "It offers competitive interest rates and flexible repayment options. "
                        "Required documents typically include identity proof, address proof, and income proof. "
                        "For specific terms, visit the official Shriram Finance website."
                    )
            return "Two-Wheeler Loans are available from Shriram Finance for purchasing motorcycles or scooters. Please check the official website for details."

        # Handle "fixed investment plan" or "fixed deposit" queries
        if "fixed investment" in query_lower or "fixed deposit" in query_lower:
            for doc in retrieved_docs:
                content_lower = doc["content"].lower()
                if "fixed deposit" in content_lower:
                    return (
                        "A Fixed Deposit from Shriram Finance is a secure investment option offering guaranteed returns at competitive interest rates. "
                        "Tenures vary, and interest can be paid monthly, quarterly, or at maturity. "
                        "For current rates and terms, visit the official Shriram Finance website."
                    )
            return "Fixed Deposits at Shriram Finance offer secure investment with competitive returns. Please check the official website for details."

        # Handle "gold loan" queries
        if "gold loan" in query_lower and "documents" not in query_lower:
            for doc in retrieved_docs:
                content_lower = doc["content"].lower()
                if "gold loan" in content_lower:
                    return (
                        "A Gold Loan from Shriram Finance allows you to borrow money by pledging gold ornaments as collateral. "
                        "It offers quick disbursal, competitive interest rates, and flexible repayment options. "
                        "Required documents include identity proof, address proof, and photos. "
                        "For more details, visit the official Shriram Finance website."
                    )
            return "Gold Loans at Shriram Finance are secured loans against gold ornaments. Please check the official website for details."

        # Fallback for unrecognized queries
        return "No specific information found for your query. Please check the official Shriram Finance website for more details."

    def generate_response(self, query, retrieved_docs):
        summary = self.summarize_context(retrieved_docs, query)
        return summary

    def answer_query(self, query):
        retrieved_docs = self.retrieve(query)
        return self.generate_response(query, retrieved_docs)

if __name__ == "__main__":
    config = load_config()
    rag = RAGModel(config)
    queries = [
        "What is the interest rate of fixed deposit",
        "What are the documents required for gold loan",
        "What are the services offered by <companyname>",
        "What is two-wheeler loan",
        "What is fixed investment plan",
        "What is gold loan"
    ]
    for query in queries:
        response = rag.answer_query(query)
        print(f"Query: {query}\nResponse: {response}\n")