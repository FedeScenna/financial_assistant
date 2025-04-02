"""
Helper Functions

This module provides utility functions for the financial assistant.
"""

import os
import sys
import traceback
import torch
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st


def check_ollama_server():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            return True, available_models
        return False, []
    except requests.exceptions.ConnectionError:
        return False, []


def check_gpu():
    """Check if GPU is available and return information about it."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        return True, f"{device_name} (CUDA {cuda_version})"
    return False, "Not available"


def get_financial_prompt_template():
    """Get the financial prompt template for the RAG system."""
    financial_template = """
You are an expert financial assistant specializing in providing concise and accurate answers about companies in the S&P 500 index. Use the provided context to generate responses. If the context does not contain relevant information, state that you do not have enough data rather than making assumptions. Ensure responses are factual, clear, and to the point.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=financial_template,
        input_variables=["context", "chat_history", "question"]
    )


def is_stock_query(query):
    """Determine if a query is related to stock data."""
    query_lower = query.lower()
    stock_keywords = [
        "stock", "price", "ticker", "share", "market", "trading", "investor", 
        "dividend", "return", "performance", "technical", "indicator", "rsi", 
        "macd", "bollinger", "volatility", "volume", "compare"
    ]
    
    # Check for stock ticker patterns (all caps 1-5 letters)
    words = query.split()
    has_ticker = any(word.isupper() and word.isalpha() and 1 <= len(word) <= 5 for word in words)
    
    # Check for stock-related keywords
    has_keywords = any(keyword in query_lower for keyword in stock_keywords)
    
    return has_ticker or has_keywords 