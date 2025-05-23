import streamlit as st
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re # Import the re module for regular expressions
import torch # Import torch
import asyncio
from requests.exceptions import ChunkedEncodingError
import logging
import sys
import time
import traceback

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.classes.__path__ = [] # Add the suggested workaround for torch path error

from src.data_loader import create_knowledge_base, load_knowledge_base, update_knowledge_base # Import update_knowledge_base
# from langchain_ollama import ChatOllama # Keep LLM import here
from langchain_openai import ChatOpenAI # Import ChatOpenAI for llama_cpp.server integration

# Initialize session state for messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("My Knowledge Base Helper")
st.write("Welcome to your personal study helper!")

PERSIST_DIRECTORY = "./chroma_db"

# Input for the knowledge base directory
kb_directory = st.text_input("Enter the path to your knowledge base directory (e.g., where your markdown notes are):")

# Button to load or create knowledge base
# Changed button text to reflect loading from storage or creating
load_create_button = st.button("Load or Create/Update Knowledge Base") # Updated button text

# Use Streamlit session state to persist the vector store
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Logic for loading or creating/updating the knowledge base
if load_create_button:
    if kb_directory:
        if os.path.isdir(kb_directory):
            # Try loading the knowledge base first
            st.info(f"Attempting to load knowledge base from {PERSIST_DIRECTORY}...")
            vectorstore = load_knowledge_base(PERSIST_DIRECTORY)

            if vectorstore is not None:
                # If loading is successful, attempt to update it
                st.success(f"Knowledge base loaded successfully from {PERSIST_DIRECTORY}. Checking for updates...")
                try:
                    updated_vectorstore = update_knowledge_base(kb_directory, PERSIST_DIRECTORY)
                    if updated_vectorstore is not None:
                        st.session_state.vectorstore = updated_vectorstore
                        st.success("Knowledge base updated.")
                    else:
                        st.warning("Update process completed, but the vector store is None.") # Should not happen if update is successful
                except ValueError as e:
                     st.error(f"Error during knowledge base update: {e}")
                     st.session_state.vectorstore = vectorstore # Keep the loaded vectorstore if update fails

            else:
                # If loading fails, create a new one
                st.warning(f"No existing knowledge base found at {PERSIST_DIRECTORY}. Creating a new one...")
                try:
                    vectorstore = create_knowledge_base(kb_directory, PERSIST_DIRECTORY)
                    if vectorstore is not None:
                         st.session_state.vectorstore = vectorstore
                         st.success("New knowledge base created and saved.")
                    else:
                         st.error("Knowledge base creation failed.")
                except ValueError as e:
                    st.error(f"Error creating knowledge base: {e}")

        else:
            st.error("The provided path is not a valid directory.")
    else:
        st.warning("Please enter a directory path.")

# --- User Query ---
st.subheader("Ask a question about your notes:")
query = st.text_input("Enter your query here:")
ask_button = st.button("Get Answer")

if ask_button:
    if query:
        if st.session_state.vectorstore:
            st.write(f"Searching for relevant information for: {query}")
            # Perform similarity search
            # We'll retrieve the top 3 most relevant documents as an example
            retrieved_docs = st.session_state.vectorstore.similarity_search(query, k=3)

            if retrieved_docs:
                st.write("Found relevant information:")
                for i, doc in enumerate(retrieved_docs):
                    # Determine page info separately to avoid f-string syntax issues
                    if 'page' in doc.metadata:
                        page_info = f", Page: {doc.metadata['page']}"
                    else:
                        page_info = ''
                    st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}{page_info}):**") # Include source and page info
                    st.markdown(doc.page_content[:500] + '...') # Display snippet using markdown
                    st.write("----")

                # --- Generating Answer with LLM ---
                st.write("Generating answer using local LLM...")
                # Initialize the local LLM using the llama_cpp.server OpenAI-compatible API
                try:
                    # Ensure the llama_cpp.server is running at http://localhost:8000
                    # Explicitly adding /v1 to the API base URL
                    llm = ChatOpenAI(
                        openai_api_base="http://localhost:8000/v1",
                        openai_api_key="not-needed",  # Required but unused
                        model="llama-2-7b.Q5_K_S.gguf",
                        stream=False, # Set stream to False
                        temperature=0.7,
                        max_tokens=1024,
                        request_timeout=600
                    )
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    # --- Generating and Displaying Answer with LLM (Non-Streaming) ---
                    st.subheader("Answer:")

                    try:
                        st.info("Connecting to LLM server and waiting for response...")
                        print(f"DEBUG: Attempting to connect to {llm.openai_api_base}")

                        prompt = [
                            {"role": "system", "content": "You explain concepts clearly in a friendly tone"},
                            {"role": "user", "content": f"Context: {context_text[:1000]}\nQuestion: {query}\nAnswer:"}
                        ]
                        print(f"Full prompt structure:\n{prompt}")

                        with st.chat_message("assistant"):
                            try:
                                print("Attempting to get full response...")
                                # Use the invoke method to get the full response
                                final_response = llm.invoke(prompt) # Use the same prompt format
                                if final_response.content:
                                    # Display the full response at once
                                    st.markdown(final_response.content)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": final_response.content
                                    })
                                    print("Full response received and displayed.")
                                else:
                                    st.error("Received empty response from LLM.")
                                    print("Invoke call returned empty content.")
                            except Exception as e:
                                st.error(f"Error getting LLM response: {str(e)}")
                                print(f"Full error trace:\n{traceback.format_exc()}")

                    except Exception as e:
                        st.error("Error during LLM interaction: " + str(e))
                        print(f"FULL ERROR TRACE: {traceback.format_exc()}")

                except Exception as e:
                    st.error(f"Error interacting with local LLM or rendering response: {e}")

            else:
                st.info("No relevant documents found.")

        else:
            st.warning("Knowledge base not loaded. Please load/create one first.")

    else:
        st.warning("Please enter a query.")