structure of the project:
rag_chatbot/
├── app.py                    # Main entry point (Streamlit UI or CLI)
├── document_loader.py        # All logic for reading local files (PDF, DOCX, etc.)
├── web_extractor.py          # (Next step) All logic for extracting text from URLs
├── vector_store.py           # Vector database setup and retrieval logic
├── qa_chain.py               # LLM + retriever integration
├── utils.py                  # Any helper utilities (text chunking, etc.)
├── requirements.txt          # Dependencies
└── data/
    └── uploaded_files/       # Temporary folder for uploaded user documents