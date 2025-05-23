import os
import json
import time
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader # We might keep this for structure, but will use pytesseract for content
import shutil
from pdf2image import convert_from_path # Import pdf2image
import pytesseract # Import pytesseract
from PIL import Image # Import Image from Pillow

# Define the embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# File to store file timestamps
FILE_STATE_FILE = "file_state.json"

def save_file_state(file_state: dict, persist_directory: str):
    """Saves the file state (path and timestamp) to a JSON file."""
    state_path = os.path.join(persist_directory, FILE_STATE_FILE)
    with open(state_path, 'w') as f:
        json.dump(file_state, f, indent=4)
    print(f"File state saved to {state_path}")

def load_file_state(persist_directory: str) -> dict:
    """Loads the file state from a JSON file."""
    state_path = os.path.join(persist_directory, FILE_STATE_FILE)
    if os.path.exists(state_path):
        print(f"Loading file state from {state_path}")
        with open(state_path, 'r') as f:
            return json.load(f)
    print(f"File state not found at {state_path}")
    return {}

def create_knowledge_base(kb_directory: str, persist_directory: str) -> Chroma:
    """Loads markdown and PDF files (including subdirectories), creates embeddings, and builds a Chroma vector store."""
    if not os.path.isdir(kb_directory):
        raise ValueError(f"Directory not found: {kb_directory}")

    documents = []
    current_file_state = {}
    supported_extensions = ('.md', '.pdf')

    print(f"Starting knowledge base creation from {kb_directory}")
    print(f"Searching for {', '.join(supported_extensions)} files...")
    for root, _, files in os.walk(kb_directory):
        for file in files:
            if file.endswith(supported_extensions):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                current_file_state[file_path] = os.path.getmtime(file_path)
                try:
                    if file.endswith('.md'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
                    elif file.endswith('.pdf'):
                        print(f"Attempting to OCR PDF file: {file_path}")
                        try:
                            print(f"Converting {file_path} pages to images using pdf2image...")
                            # Convert PDF to images
                            pages = convert_from_path(file_path)
                            print(f"Successfully converted {len(pages)} pages to images.")
                            # Process each page image with Tesseract
                            for i, page_image in enumerate(pages):
                                print(f"  Processing page {i+1}/{len(pages)} with Tesseract OCR...")
                                text = pytesseract.image_to_string(page_image)
                                print(f"  Finished processing page {i+1}.")
                                # Create a Document for each page
                                documents.append(Document(page_content=text, metadata={"source": file_path, "page": i+1}))
                            print(f"Successfully processed PDF {file} with OCR.")
                        except pytesseract.TesseractNotFoundError:
                            print(f"Error: Tesseract is not installed or not in your PATH. Cannot process {file}.")
                            print("Please install Tesseract OCR engine.")
                            # Optionally, you could skip this file or raise an error
                            continue # Skip this file
                        except Exception as e:
                            print(f"Error processing PDF file {file_path}: {e}")
                            print(f"Please ensure poppler-utils is installed and in your PATH for pdf2image to work.")
                            # Optionally, you could skip this file or raise an error
                            continue # Skip this file
                    print(f"Finished processing file: {file}") # This line was slightly misplaced, moved it here.
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    if not documents:
        print(f"No supported files ({', '.join(supported_extensions)}) found or content read in {kb_directory} or its subdirectories.")
        # If persistence directory exists, clear it since no files were found
        if os.path.exists(persist_directory):
             print(f"No documents created, clearing persistent directory {persist_directory}")
             shutil.rmtree(persist_directory)
        return None

    print(f"Read content from {len(documents)} documents.")

    # --- Text Splitting ---
    print("Splitting text into chunks...")
    # Use a splitter that works for both markdown and plain text from PDF
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # --- Creating Embeddings and Vector Store ---
    print("Creating embeddings and building vector store...")

    # Remove existing data if directory exists to ensure fresh build
    if os.path.exists(persist_directory):
        print(f"Removing existing data in {persist_directory}")
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vectorstore.persist()

    # Save the current file state after successful creation
    save_file_state(current_file_state, persist_directory)

    print(f"Knowledge base created and saved to {persist_directory}.")
    return vectorstore

def update_knowledge_base(kb_directory: str, persist_directory: str) -> Chroma or None:
    """Updates an existing Chroma vector store based on changes in markdown and PDF files."""
    if not os.path.isdir(kb_directory):
        raise ValueError(f"Directory not found: {kb_directory}")

    if not os.path.exists(persist_directory):
        print(f"Persistent directory not found at {persist_directory}. Cannot update.")
        return None

    # Load the existing file state and vector store
    last_file_state = load_file_state(persist_directory)
    vectorstore = load_knowledge_base(persist_directory) # Load the existing vector store

    if vectorstore is None:
        print("Could not load existing knowledge base. Cannot update.")
        return None

    current_file_state = {}
    current_files_found = {} # Store file paths for easier lookup
    supported_extensions = ('.md', '.pdf')

    print(f"Starting knowledge base update from {kb_directory}")
    # Scan current files and build current state
    for root, _, files in os.walk(kb_directory):
        for file in files:
            if file.endswith(supported_extensions):
                file_path = os.path.join(root, file)
                current_file_state[file_path] = os.path.getmtime(file_path)
                current_files_found[file_path] = True

    # Identify added, modified, and deleted files
    added_files = [f for f in current_file_state if f not in last_file_state]
    modified_files = [f for f in current_file_state if f in last_file_state and current_file_state[f] > last_file_state[f]]
    deleted_files = [f for f in last_file_state if f not in current_files_found]

    if not added_files and not modified_files and not deleted_files:
        print("No changes detected in supported files.")
        return vectorstore # Return the existing vector store

    print(f"Detected changes: Added={len(added_files)}, Modified={len(modified_files)}, Deleted={len(deleted_files)}")

    # Process deleted files
    if deleted_files:
        print("Processing deleted files...")
        for file_path in deleted_files:
             print(f"Detected deleted file: {file_path}")
        # Note: Deleting documents from Chroma by source file path directly is complex.
        # A full re-index of the affected files or a different ChromaDB strategy is needed for robust deletion handling.
        # For this example, we will simply note the deletion.
        print("Note: For complete removal of data from deleted files, a full knowledge base re-creation is recommended for this example setup.")

        # A more advanced approach would involve querying Chroma for document IDs associated with deleted file paths
        # and using vectorstore.delete(ids=...). This requires more intricate Chroma interaction.

    # Process added and modified files
    files_to_process = added_files + modified_files
    if files_to_process:
        print("Processing added and modified files...")
        documents_to_add = []
        for file_path in files_to_process:
             print(f"Loading file for update: {file_path}")
             try:
                 if file_path.endswith('.md'):
                     with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents_to_add.append(Document(page_content=content, metadata={"source": file_path}))
                 elif file_path.endswith('.pdf'):
                     print(f"Attempting to OCR PDF file for update: {file_path}")
                     try:
                        print(f"Converting {file_path} pages to images using pdf2image for update...")
                        # Convert PDF to images
                        pages = convert_from_path(file_path)
                        print(f"Successfully converted {len(pages)} pages to images for update.")
                        # Process each page image with Tesseract
                        for i, page_image in enumerate(pages):
                            print(f"  Processing page {i+1}/{len(pages)} with Tesseract OCR for update...")
                            text = pytesseract.image_to_string(page_image)
                            print(f"  Finished processing page {i+1} for update.")
                            # Create a Document for each page
                            documents_to_add.append(Document(page_content=text, metadata={"source": file_path, "page": i+1}))
                        print(f"Successfully processed PDF {os.path.basename(file_path)} with OCR for update.")
                     except pytesseract.TesseractNotFoundError:
                        print(f"Error: Tesseract is not installed or not in your PATH. Cannot process {file_path} for update.")
                        print("Please install Tesseract OCR engine.")
                        # Optionally, you could skip this file or raise an error
                        continue # Skip this file
                     except Exception as e:
                        print(f"Error processing PDF file {file_path} for update: {e}")
                        print(f"Please ensure poppler-utils is installed and in your PATH for pdf2image to work.")
                        # Optionally, you could skip this file or raise an error
                        continue # Skip this file
                 print(f"Finished processing file for update: {os.path.basename(file_path)}") # This line was slightly misplaced, moved it here.
             except Exception as e:
                print(f"Error reading file {file_path} for update: {e}")

        if documents_to_add:
            # For modified files, you might need to delete existing entries before adding.
            # Again, without easy ID access via source metadata, adding new documents with the same source
            # effectively updates the information available for that source, though old chunks might remain.
            # A robust solution requires explicit deletion by ID.
            print("Adding/updating documents for added/modified files...")
            # Using a splitter that works for both markdown and plain text from PDF
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts_to_add = text_splitter.split_documents(documents_to_add)

            # Add the new/modified documents to the vector store
            vectorstore.add_documents(texts_to_add)

            # Persist the changes
            vectorstore.persist()
            print(f"Added/updated {len(texts_to_add)} chunks to the knowledge base.")
        else:
             print("No content to add/update from added/modified files.")

    # Save the updated file state
    save_file_state(current_file_state, persist_directory)
    print("Updated file state saved.")

    # Note: Deletion handling is basic. Full re-creation might be needed after significant deletions.
    if deleted_files:
         print("Please note: Data from deleted files may still be in the knowledge base. Consider re-creating if accurate deletion is critical.")

    return vectorstore

def load_knowledge_base(persist_directory: str) -> Chroma or None:
    """Loads a Chroma vector store from a local directory."""
    if not os.path.isdir(persist_directory) or not os.path.exists(os.path.join(persist_directory, FILE_STATE_FILE)):
        print(f"Persistent directory not found or missing file state: {persist_directory}")
        return None

    print(f"Loading knowledge base from {persist_directory}...")
    try:
        # We need the embedding function to load the Chroma collection correctly
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Knowledge base loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error loading knowledge base from {persist_directory}: {e}")
        return None
