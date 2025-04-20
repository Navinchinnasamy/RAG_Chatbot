import faiss
import numpy as np
import os
from modules.config_loader import load_config

class VectorStore:
    def __init__(self, config):
        self.embedding_dir = config["embeddings"]["output_dir"]
        self.output_dir = config["vector_store"]["output_dir"]
        self.index_path = config["vector_store"]["index_path"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.index = None
        self.doc_ids = []

    def build(self):
        try:
            embeddings = np.load(os.path.join(self.embedding_dir, "embeddings.npy"), allow_pickle=True)
            dimension = embeddings[0]["embedding"].shape[0]
            self.index = faiss.IndexFlatL2(dimension)
            vectors = np.array([item["embedding"] for item in embeddings], dtype=np.float32)
            self.doc_ids = [item["id"] for item in embeddings]
            self.index.add(vectors)
            try:
                faiss.write_index(self.index, self.index_path)
                print(f"Vector store built and saved to {self.index_path}")
            except Exception as e:
                print(f"Error writing index to {self.index_path}: {e}")
                print("Please check file permissions, ensure the file is not locked, or run the script as Administrator.")
                raise
        except FileNotFoundError:
            print(f"Error: embeddings.npy not found in {self.embedding_dir}. Run embedding_generator.py first.")
            raise
        except Exception as e:
            print(f"Error building vector store: {e}")
            raise

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            embeddings = np.load(os.path.join(self.embedding_dir, "embeddings.npy"), allow_pickle=True)
            self.doc_ids = [item["id"] for item in embeddings]
        else:
            print(f"Index file {self.index_path} not found. Building new index...")
            self.build()

    def search(self, query_embedding, k):
        if self.index is None:
            self.load()
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], distance))
        return results

if __name__ == "__main__":
    config = load_config()
    vector_store = VectorStore(config)
    vector_store.build()