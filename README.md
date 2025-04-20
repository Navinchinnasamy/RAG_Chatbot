pip install -r requirements.txt

python -m modules.web_scraper
python -m modules.knowledgebase_builder
python -m modules.embedding_generator
python -m modules.vector_store

streamlit run streamlit_app.py --server.fileWatcherType=none
streamlit run streamlit_app.py --logger.level=error --server.fileWatcherType=none


python flask_api.py


website_scraper_rag/
├── config.yml
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── web_scraper.py
│   ├── knowledgebase_builder.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── rag_model.py
│   ├── context_builder.py
│   ├── chatbot.py
├── streamlit_app.py
├── flask_api.py
├── data/
│   ├── raw/
│   ├── knowledgebase/
│   ├── embeddings/
│   ├── vectorstore/
└── README.md