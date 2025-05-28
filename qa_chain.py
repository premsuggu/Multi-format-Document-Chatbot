# qa_chain.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

from utils import chunk_text, clean_text 


class QAChain:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=100, overlap=20):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, full_text: str):
        """
        Clean, chunk, embed, and build FAISS index for similarity search.
        """
        cleaned_text = clean_text(full_text)
        self.chunks = chunk_text(cleaned_text, self.chunk_size, self.overlap)

        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, top_k=10) -> list[str]:
        """
        Search FAISS index for top_k similar chunks given a query.
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0]]

    def answer_question(self, question: str, contexts: list[str], model_name="llama3.2") -> str:
        """
        Use Ollama API to generate answer from retrieved contexts.
        """
        context_text = "\n\n".join(contexts)
        prompt = f"""Answer the following question using ONLY the context provided. If the answer isn't present, say "I don't know."

Context:
{context_text}

Question: {question}

Answer:"""
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response']

    def chat(self, question: str, contexts: list[str] = None, chat_history: list = None, model_name="llama3.2"):
        """
        Chat interface to Ollama with context and chat history.
        
        Returns response text and updated chat history.
        """
        if chat_history is None:
            chat_history = []

        system_message = "You are a helpful assistant."
        if contexts:
            context_text = "\n\n".join(contexts)
            system_message = f"""You are a helpful assistant. Answer the following question using the context provided. 
If the answer isn't present in the context, say "I don't know."

Context:
{context_text}
"""

        if not chat_history:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]
        else:
            messages = [{"role": "system", "content": system_message}]
            messages.extend(chat_history)
            messages.append({"role": "user", "content": question})

        response = ollama.chat(model=model_name, messages=messages)
        response_text = response['message']['content']

        updated_history = chat_history.copy()
        updated_history.append({"role": "user", "content": question})
        updated_history.append({"role": "assistant", "content": response_text})

        # Keep last 10 messages max (5 user-assistant exchanges)
        if len(updated_history) > 10:
            updated_history = updated_history[-10:]

        return response_text, updated_history
