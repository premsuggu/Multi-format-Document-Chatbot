# Sush: AI-Powered Document Question Answering System

A general-purpose AI system to extract, index, and answer questions from multi-format documents using LLMs, OCR, semantic search, and structured data handling.

## Overview

Sush is an intelligent, modular system built to automatically ingest and understand documents of various formats—PDFs, Word files, Excel sheets, presentations, images, and more—and provide accurate, context-aware answers to natural language queries. Designed to mirror real-world enterprise needs, Sush integrates state-of-the-art AI techniques to unify structured and unstructured data into a seamless, searchable knowledge base.

Developed as part of a self project, this project showcases how modern NLP, document parsing, and data science pipelines can work together to transform information retrieval.

## Features

- **Multi-Format File Support**
  - Accepts and parses PDFs, DOCX, PPTX, XLSX, CSV, JSON, PNG, JPG, and TXT files
  - Extracts text, tables, and metadata using robust parsers (e.g., `PyMuPDF`, `python-docx`, `pdf2image`, `openpyxl`)

- **Natural Language Question Answering**
  - Combines information retrieval + LLM-based response generation
  - Answers factual and analytical questions grounded in document content

- **Reference Link Crawling**
  - Automatically fetches and incorporates the text from cited web URLs in documents

- **OCR for Image & Scanned Documents**
  - Integrates `Tesseract OCR` to extract text from images and scanned PDFs.
  - Falls back to OCR when native text extraction fails.

- **Semantic Search with Embeddings**
  - Utilizes pre-trained sentence embeddings (`Sentence Transformers` or OpenAI API).
  - Enables vector-based similarity search using `FAISS` or `ChromaDB`.

- **Structured Data Intelligence**
  - Parses tables and structured files using `pandas`.
  - Supports basic analytics (e.g., totals, filtering) in response to user queries.

- **Modular, Scalable Pipeline**
  - Clear separation of ingestion, processing, retrieval, and response modules.
  - Easy to extend with new file types, models, or APIs.

## Tech Stack

| Component         | Tools Used |
|------------------|------------|
| LLM                            | llama3.2 and anyother open-sourced model |
| File Parsing                   | `PyMuPDF`, `python-docx`, `openpyxl`, `python-pptx` |
| OCR                            | `Tesseract OCR` via `pytesseract` |
| Semantic Search                | `FAISS`, `sentence-transformers` |
| Web Crawling                   | `requests`, `BeautifulSoup` |
| Data Analysis                  | `pandas`, `numpy` |
| Frontend (yet to be developed) | Streamlit / Flask / Gradio (modular setup) |
