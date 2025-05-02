"""
RAG Engine

This module provides the core RAG functionality for the financial assistant.
"""

import os
import traceback
import streamlit as st
import torch
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

def check_ollama_model(model_name):
    """
    Check if the specified Ollama model exists.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model exists, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return model_name in models
        return False
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama server. Make sure Ollama is running.")
        return False
    except Exception as e:
        st.error(f"Error checking Ollama model: {str(e)}")
        return False

def get_financial_prompt_template():
    """
    Returns a prompt template for financial analysis.
    """
    template = """You are a helpful financial assistant. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def load_vector_store(embeddings_dir, model_name="llama3.1:8b"):
    """
    Load the vector store from disk.
    
    Args:
        embeddings_dir (str): Path to the directory containing the embeddings
        model_name (str): Name of the model to use for embeddings
        
    Returns:
        Chroma: The loaded vector store
    """
    try:
        # Check if model exists
        if not check_ollama_model(model_name):
            raise ValueError(f"Model {model_name} not found in Ollama. Please pull it first.")
            
        # Initialize embeddings with the specified model
        embeddings = OllamaEmbeddings(model=model_name)
        
        # Configure Chroma to use GPU if available
        if os.environ.get("CHROMA_GPU") == "1" and torch.cuda.is_available():
            st.info("Using GPU for vector store operations")
            # Chroma will automatically use GPU if available and CHROMA_GPU=1
            vector_store = Chroma(
                persist_directory=embeddings_dir,
                embedding_function=embeddings
            )
        else:
            st.info("Using CPU for vector store operations")
            vector_store = Chroma(
                persist_directory=embeddings_dir,
                embedding_function=embeddings
            )
            
        return vector_store
    except Exception as e:
        st.session_state.debug_info = f"Error in load_vector_store: {str(e)}\n{traceback.format_exc()}"
        raise

def get_conversation_chain(vector_store, model_name="llama3.1:8b", temperature=0.7):
    """
    Create a conversation chain for RAG.
    
    Args:
        vector_store (Chroma): The vector store to use for retrieval
        model_name (str): Name of the model to use for generation
        temperature (float): Temperature parameter for generation
        
    Returns:
        ConversationalRetrievalChain: The conversation chain
    """
    try:
        # Check if model exists
        if not check_ollama_model(model_name):
            raise ValueError(f"Model {model_name} not found in Ollama. Please pull it first.")
            
        # Initialize Ollama with the selected model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Get the financial prompt template
        PROMPT = get_financial_prompt_template()

        # Configure Ollama to use GPU if available
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.is_available():
            st.info("Using GPU for model inference")
            llm = Ollama(
                model=model_name, 
                temperature=temperature,
                callback_manager=callback_manager,
                num_gpu=1  # Use GPU for inference
            )
        else:
            st.info("Using CPU for model inference")
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
            verbose=True
        )
        return conversation_chain
    except Exception as e:
        st.session_state.debug_info = f"Error in get_conversation_chain: {str(e)}\n{traceback.format_exc()}"
        raise 