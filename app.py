import os
from document_loader import extract_txt_content
from web_extractor import extract_web_content, extract_urls
from qa_chain import QAChain

def main():
    print("Starting Document QA Application...")

    # Step 1: Ask user to choose the model
    available_models = ['llama3', 'mistral']
    model_name = input(f"Choose model {available_models}: ").strip().lower()

    if model_name not in available_models:
        print(f"Invalid choice. Defaulting to 'llama3'.")
        model_name = 'llama3'

    # Initial files to load
    files = [r"data\image.png"]

    # Extract text content from documents
    print("Extracting text from files...")
    text_content = extract_txt_content(files)
    print("Extracting web content from URLs in text...")
    urls = extract_urls(text_content)
    web_contents = []
    for url in urls:
        try:
            web_contents.append(extract_web_content(url))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    web_content = "\n".join(web_contents)
    all_content = text_content + "\n" + web_content

    # Initialize QA system
    qa = QAChain()
    print("Building vector store index...")
    qa.build_index(all_content)

    chat_history = []

    print("You can now ask questions! Type 'exit' to quit, 'upload' to add new files.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        elif query.lower() == "upload":
            path = input("Enter file path to upload: ").strip()
            if os.path.exists(path):
                files.append(path)
                print(f"Processing {path} ...")
                text_content = extract_txt_content(files)
                web_content = extract_web_content(text_content)
                all_content = text_content + "\n" + web_content
                qa.build_index(all_content)
                print("File added and index rebuilt!")
            else:
                print("File not found. Please try again.")

        else:
            # Search and answer using selected model
            relevant_chunks = qa.search(query, top_k=5)
            answer, chat_history = qa.chat(query, relevant_chunks, chat_history, model_name=model_name)
            print(f"\n Bot ({model_name}): {answer}")

if __name__ == "__main__":
    main()
