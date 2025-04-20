from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
from modules.config_loader import load_config

class EmbeddingGenerator:
    def __init__(self, config):
        self.model_name = config["embeddings"]["model"]
        self.model = SentenceTransformer(self.model_name)
        self.input_dir = config["knowledgebase"]["output_dir"]
        self.output_dir = config["embeddings"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self):
        with open(os.path.join(self.input_dir, "knowledgebase.json"), "r") as f:
            knowledgebase = json.load(f)
        
        embeddings = []
        for item in knowledgebase:
            embedding = self.model.encode(item["content"], show_progress_bar=False)
            embeddings.append({"id": item["id"], "embedding": embedding})
        
        # Save embeddings
        np.save(os.path.join(self.output_dir, "embeddings.npy"), embeddings)
        return embeddings

if __name__ == "__main__":
    config = load_config()
    generator = EmbeddingGenerator(config)
    generator.generate()