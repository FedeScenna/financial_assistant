"""
Streamlit RAG Application

This module provides a Streamlit interface for the RAG-based financial assistant.
"""

import os
import streamlit as st
import torch
import requests
from engine import load_vector_store, get_conversation_chain, check_ollama_model

# Set page config
st.set_page_config(
    page_title="Financial Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = ""

def check_gpu_availability():
    """Check if GPU is available and return device info"""
    if torch.cuda.is_available():
        return {
            "available": True,
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count()
        }
    return {
        "available": False,
        "device": "cpu"
    }

def pull_ollama_model(model_name):
    """Pull the specified Ollama model"""
    try:
        response = requests.post(
            f"http://localhost:11434/api/pull",
            json={"name": model_name}
        )
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama server. Make sure Ollama is running.")
        return False
    except Exception as e:
        st.error(f"Error pulling model: {str(e)}")
        return False

def main():
    st.title("Financial Assistant 💰")
    st.markdown("""
    Welcome to the Financial Assistant! Ask me questions about financial data and I'll help you analyze it.
    """)

    # Check GPU availability
    gpu_info = check_gpu_availability()
    if gpu_info["available"]:
        st.success(f"✅ GPU Available: {gpu_info['device_name']} (Device count: {gpu_info['device_count']})")
    else:
        st.warning("⚠️ No GPU detected. Running on CPU.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # GPU settings
        st.subheader("GPU Settings")
        use_gpu = st.checkbox(
            "Use GPU (if available)",
            value=gpu_info["available"],
            disabled=not gpu_info["available"]
        )
        
        # Model settings
        st.subheader("Model Settings")
        embeddings_dir = st.text_input(
            "Embeddings Directory",
            value="embeddings",
            help="Path to the directory containing the vector embeddings"
        )
        model_name = st.selectbox(
            "Model",
            ["llama3.1:8b"],
            index=0,
            help="Select the model to use for generation"
        )

        # Check if model exists
        model_exists = check_ollama_model(model_name)
        if not model_exists:
            st.warning(f"Model {model_name} not found in Ollama.")
            if st.button("Pull Model"):
                with st.spinner(f"Pulling {model_name}..."):
                    if pull_ollama_model(model_name):
                        st.success(f"Successfully pulled {model_name}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to pull {model_name}")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random, lower values make it more deterministic"
        )

        if st.button("Initialize System"):
            with st.spinner("Loading vector store and initializing conversation chain..."):
                try:
                    # Set environment variables for GPU usage
                    if use_gpu and gpu_info["available"]:
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        os.environ["CHROMA_GPU"] = "1"
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        os.environ["CHROMA_GPU"] = "0"

                    # Check if model exists before initializing
                    if not check_ollama_model(model_name):
                        raise ValueError(f"Model {model_name} not found in Ollama. Please pull it first.")

                    vector_store = load_vector_store(embeddings_dir, model_name)
                    st.session_state.conversation = get_conversation_chain(
                        vector_store,
                        model_name=model_name,
                        temperature=temperature
                    )
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
                    st.error(st.session_state.debug_info)

    # Main chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    # Chat input
    if prompt := st.chat_input("Ask a question about financial data"):
        if st.session_state.conversation is None:
            st.error("Please initialize the system first using the sidebar.")
            return

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    sources = [doc.metadata.get("source", "Unknown source") for doc in response["source_documents"]]
                    
                    st.markdown(answer)
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
                    st.error(st.session_state.debug_info)

if __name__ == "__main__":
    main() 