# RAG_POC

## Overview
**RAG_POC** is a Retrieval-Augmented Generation (RAG) chatbot that answers queries about financial services by scraping content from a specified website, processing it to remove noise (e.g., headers, footers), and retrieving relevant information using a vector store (FAISS) with embeddings (Sentence Transformers). It supports command-line, Streamlit web, and Flask API interfaces.

### Features
- **Web Scraping**: Extracts clean content, excluding menus and buttons.
- **Knowledgebase**: Builds a structured knowledgebase from scraped data.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for vector representations.
- **Retrieval**: Employs FAISS for efficient document retrieval.
- **Query Handling**: Answers queries about services, loan documents, and investment plans.
- **Interfaces**: Command-line, Streamlit app, Flask API.

## Prerequisites
- **Python**: 3.8 or higher.
- **Virtual Environment**: Recommended.
- **Dependencies**:
- `requests`
- `beautifulsoup4`
- `tqdm`
- `faiss-cpu`
- `sentence-transformers`
- `numpy`
- `streamlit`
- `flask`
- `pyyaml`

## Setup Instructions
1. **Clone the Repository**:
```bash
git clone <repository-url>
cd RAG_POC
```

2. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows
```

3. **Install Dependencies**:
- Create `requirements.txt`:
    ```
    requests
    beautifulsoup4
    tqdm
    faiss-cpu
    sentence-transformers
    numpy
    streamlit
    flask
    pyyaml
    ```
- Install:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure**:
- Edit `config.yml` to set the target website's base domain and page URLs for scraping. Example:
    ```yaml
    scraper:
    base_domain: "<website-url>"
    pages:
        urls:
        - "<website-url>/"
        - "<website-url>/about"
        - "<website-url>/products"
        max_pages: 1000
    articles:
        urls: []
        max_articles: 1000
    output_dir: "data/raw"
    embeddings:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: "data/embeddings"
    rag:
    chunk_size: 1000
    top_k: 3
    chatbot:
    context_window: 5
    knowledgebase:
    output_dir: "data/knowledgebase"
    vector_store:
    output_dir: "data/vectorstore"
    index_path: "data/vectorstore/index.faiss"
    ```

5. **Run the Pipeline**:
```bash
python modules/web_scraper.py
python modules/knowledgebase_builder.py
python modules/embedding_generator.py
python modules/vector_store.py
```

## Usage
### Command-Line
Test the chatbot:
```bash
python modules/rag_model.py
```
**Example Queries**:
- "What are the documents required for a loan?"
- "What services are offered?"
- "What is a two-wheeler loan?"

### Streamlit Web App
```bash
streamlit run streamlit_app.py
```
- Open `http://localhost:8501` and enter queries.

### Flask API
```bash
python flask_api.py
```
- Send POST requests:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query":"What are the services offered?"}' http://localhost:5000/chat
```

## Project Structure
```
RAG_POC/
├── config.yml               # Configuration file
├── requirements.txt         # Dependencies
├── streamlit_app.py         # Streamlit interface
├── flask_api.py             # Flask API
├── modules/
│   ├── web_scraper.py       # Scrapes website content
│   ├── knowledgebase_builder.py  # Builds knowledgebase
│   ├── embedding_generator.py    # Generates embeddings
│   ├── vector_store.py      # Builds FAISS store
│   ├── rag_model.py         # RAG model
│   ├── config_loader.py     # Loads config.yml
├── data/
│   ├── raw/                 # Scraped content
│   ├── knowledgebase/       # knowledgebase.json
│   ├── embeddings/          # embeddings.npy
│   ├── vectorstore/         # index.faiss
├── venv/                    # Virtual environment
└── README.md                # This file
```

## Troubleshooting
1. **Permission Error for `index.faiss`**:
- Ensure no process locks `data/vectorstore/index.faiss`.
- Grant write permissions to `data/vectorstore`.
- Run as Administrator (Windows) or with `sudo` (Linux/Mac).
- Delete existing file:
    ```bash
    rm data/vectorstore/index.faiss
    ```

2. **Noisy Responses**:
- Check `data/raw/` for headers/footers.
- Update `web_scraper.py` to exclude site-specific noise.

3. **Incorrect Responses**:
- Verify `config.yml` URLs and `knowledgebase.json` content.
- Check `rag_model.py` debug logs for retrieved documents.

## Contributing
- Submit issues or pull requests via the repository.
- Ensure changes include tests and align with the project structure.

## License
For educational purposes. Ensure compliance with the target website’s terms of use for scraping.
"""


-- streamlit run streamlit_app.py --logger.level=error --server.fileWatcherType=none
---
