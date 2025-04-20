import os
from modules.config_loader import load_config

class KnowledgeBaseBuilder:
    def __init__(self, config):
        self.input_dir = config["scraper"]["output_dir"]
        self.output_dir = config["knowledgebase"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def build(self):
        knowledgebase = []
        for filename in os.listdir(self.input_dir):
            with open(os.path.join(self.input_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                knowledgebase.append({"id": filename, "content": content})
        
        # Save knowledgebase as JSON
        import json
        with open(os.path.join(self.output_dir, "knowledgebase.json"), "w") as f:
            json.dump(knowledgebase, f)
        return knowledgebase

if __name__ == "__main__":
    config = load_config()
    builder = KnowledgeBaseBuilder(config)
    builder.build()