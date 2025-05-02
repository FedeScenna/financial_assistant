"""
RAG Engine

This module provides the core RAG functionality for the financial assistant.
"""

import os
import pickle
import traceback
import torch
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st

from src.utils.helpers import get_financial_prompt_template


def load_vector_store(embeddings_dir, model_name="llama3:8b"):
    """
    Load the vector store from disk.
    
    Args:
        embeddings_dir (str): Path to the directory containing the embeddings
        model_name (str): Name of the model to use for embeddings
        
    Returns:
        FAISS: The loaded vector store
    """
    try:
        # Important: We need to use the SAME embedding model that was used to create the embeddings
        # This is critical to avoid dimension mismatch errors
        
        # First try to load the index to determine what embedding model was used
        # Try to find the original model used for embeddings
        try:
            # Check if there's a metadata file that might contain model info
            metadata_path = os.path.join(embeddings_dir, "index_metadata.pickle")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    if "model" in metadata:
                        original_model = metadata["model"]
                        st.info(f"Using original embedding model: {original_model}")
                        embeddings = OllamaEmbeddings(model=original_model)
                    else:
                        # If no model info, use the model that created the embeddings
                        # This is "llama3.1:8b" based on user information
                        embeddings = OllamaEmbeddings(model="llama3.1:8b")
                        st.warning("Using default embedding model: llama3.1:8b")
            else:
                # If no metadata file, use the model that created the embeddings
                embeddings = OllamaEmbeddings(model="llama3.1:8b")
                st.warning("Using default embedding model: llama3.1:8b")
        except Exception as e:
            # If any error occurs during metadata loading, use the model that created the embeddings
            st.warning(f"Error loading embedding metadata: {str(e)}. Using default model: llama3.1:8b")
            embeddings = OllamaEmbeddings(model="llama3.1:8b")
        
        # Load the vector store with the determined embeddings
        vector_store = FAISS.load_local(embeddings_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.session_state.debug_info = f"Error in load_vector_store: {str(e)}\n{traceback.format_exc()}"
        raise


def get_conversation_chain(vector_store, model_name="llama3:8b", temperature=0.7):
    """
    Create a conversation chain for RAG.
    
    Args:
        vector_store (FAISS): The vector store to use for retrieval
        model_name (str): Name of the model to use for generation
        temperature (float): Temperature parameter for generation
        
    Returns:
        ConversationalRetrievalChain: The conversation chain
    """
    try:
        # Initialize Ollama with the selected model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Get the financial prompt template
        PROMPT = get_financial_prompt_template()

        llm = Ollama(
            model=model_name, 
            temperature=temperature,
            callback_manager=callback_manager
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True  # Enable verbose mode for debugging
        )
        return conversation_chain
    except Exception as e:
        st.session_state.debug_info = f"Error in get_conversation_chain: {str(e)}\n{traceback.format_exc()}"
        raise 