"""
Financial Assistant Application

This is the main application file for the Financial Assistant.
"""

import streamlit as st
import os
import sys
import traceback
import pandas as pd
from PIL import Image
import io
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from src.utils.helpers import check_ollama_server, check_gpu, is_stock_query
from src.rag.engine import get_conversation_chain
from src.stock_analysis.analyzer import load_stock_data, process_stock_query


def load_vector_store(embeddings_dir: str, model_name: str = "llama3.1:8b"):
    """
    Load the vector store from the embeddings directory.
    
    Args:
        embeddings_dir (str): Directory containing the embeddings
        model_name (str): Name of the model to use for embeddings
        
    Returns:
        Chroma: Loaded vector store
    """
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=model_name)
        
        # Load the Chroma database
        vector_store = Chroma(
            persist_directory=os.path.join(embeddings_dir, "chroma_db"),
            embedding_function=embeddings
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        raise


# Initialize session state variables
def init_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = ""
    if 'stock_handler' not in st.session_state:
        st.session_state.stock_handler = None
    if 'query_mode' not in st.session_state:
        st.session_state.query_mode = "rag"  # Default to RAG mode


def setup_page():
    """Set up the page configuration and title."""
    st.set_page_config(page_title="Financial Assistant", layout="wide")
    st.title("📊 Financial Assistant")
    
    # Check Ollama server
    ollama_running, available_models = check_ollama_server()
    if not ollama_running:
        st.error("⚠️ Ollama server is not running. Please start Ollama with 'ollama serve' in a terminal.")
        st.stop()
    
    # Add a note about Ollama
    has_gpu, gpu_info = check_gpu()
    gpu_status = f"🔥 GPU Acceleration: {gpu_info}" if has_gpu else "⚠️ GPU not detected, using CPU"
    
    st.info(f"This application uses Llama via Ollama for local inference. {gpu_status}")
    
    return ollama_running, available_models


def sidebar_config(available_models):
    """Configure the sidebar for data source selection and model configuration."""
    with st.sidebar:
        st.subheader("Configuration")
        
        # Tabs for different data sources
        data_source = st.radio(
            "Select Data Source",
            ["Document Embeddings", "Stock Price Data", "Both"],
            index=2,
            help="Choose which data source to use for answering questions"
        )
        
        # Document embeddings configuration
        embeddings_dir = None
        if data_source in ["Document Embeddings", "Both"]:
            st.subheader("Document Embeddings")
            
            # Get only directories in the current path
            all_items = os.listdir('.')
            directories = [item for item in all_items if os.path.isdir(item)]
            
            # Add option to enter custom path
            directories.insert(0, "Custom path")
            
            selection = st.selectbox(
                "Select a folder containing embeddings:",
                directories,
                index=0,
                help="Choose a folder where your embeddings are stored"
            )
            
            # If custom path is selected, show text input
            if selection == "Custom path":
                embeddings_dir = st.text_input(
                    "Enter the path to your embeddings directory:",
                    value="embeddings",
                    help="This should be the directory where you saved your embeddings using process_pdfs.py"
                )
            else:
                embeddings_dir = selection
                
            # Display the selected path
            st.caption(f"Selected path: {os.path.abspath(embeddings_dir)}")
        
        # Stock data configuration
        stock_data_path = None
        if data_source in ["Stock Price Data", "Both"]:
            st.subheader("Stock Price Data")
            
            stock_data_path = st.text_input(
                "Path to stock price CSV file:",
                value="data/price_data.csv",
                help="Path to the CSV file containing stock price history"
            )
        
        # Model configuration
        st.subheader("Model Configuration")
        
        # Show available models from Ollama
        if available_models:
            model_options = available_models
        else:
            model_options = ["llama3.1:8b", "llama3:70b", "llama2", "mistral"]
        
        model_name = st.selectbox(
            "Select Ollama Model:",
            model_options,
            index=0,
            help="Select the model to use for answering questions. Make sure it's available in your Ollama installation."
        )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Load data button
        load_button = st.button("Load Data")
        
        return data_source, embeddings_dir, stock_data_path, model_name, temperature, load_button


def load_data(data_source, embeddings_dir, stock_data_path, model_name, temperature):
    """Load data based on the selected data source."""
    # Load document embeddings if selected
    if data_source in ["Document Embeddings", "Both"]:
        if os.path.exists(embeddings_dir):
            try:
                vector_store = load_vector_store(embeddings_dir, model_name)
                # Update the conversation with the selected model and temperature
                st.session_state.conversation = get_conversation_chain(
                    vector_store, 
                    model_name=model_name, 
                    temperature=temperature
                )
                st.success("Embeddings loaded successfully!")
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                with st.expander("Debug Information"):
                    st.code(st.session_state.debug_info)
        else:
            st.error(f"Directory {embeddings_dir} does not exist. Please process your documents first using process_pdfs.py")
    
    # Load stock data if selected
    if data_source in ["Stock Price Data", "Both"]:
        if os.path.exists(stock_data_path):
            try:
                st.session_state.stock_handler = load_stock_data(stock_data_path)
                if st.session_state.stock_handler:
                    st.success(f"Stock data loaded successfully! Found {len(st.session_state.stock_handler.tickers)} tickers.")
            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")
        else:
            st.error(f"File {stock_data_path} does not exist.")


def chat_interface():
    """Create the chat interface for user interaction."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("Chat with Financial Assistant")
        
        # Check if any data is loaded
        data_loaded = (st.session_state.conversation is not None or st.session_state.stock_handler is not None)
        
        if data_loaded:
            user_question = st.chat_input("Ask a question about finances, companies, or stock prices...")
            if user_question:
                # Determine if this is a stock-related query
                is_stock = is_stock_query(user_question)
                
                with st.spinner("Thinking... (This might take a moment as processing is done locally)"):
                    try:
                        # Process stock query
                        if is_stock and st.session_state.stock_handler:
                            response_text = process_stock_query(user_question, st.session_state.stock_handler)
                            st.session_state.chat_history.append(("user", user_question))
                            st.session_state.chat_history.append(("assistant", response_text))
                            st.session_state.query_mode = "stock"
                        
                        # Process RAG query
                        elif st.session_state.conversation:
                            response = st.session_state.conversation({
                                'question': user_question
                            })
                            st.session_state.chat_history.append(("user", user_question))
                            st.session_state.chat_history.append(("assistant", response['answer']))
                            st.session_state.query_mode = "rag"
                            
                            # Store source documents for display
                            if 'source_documents' in response and response['source_documents']:
                                st.session_state.last_sources = response['source_documents']
                        
                        # No appropriate handler
                        else:
                            if is_stock:
                                error_msg = "Stock data is not loaded. Please load stock data to answer stock-related questions."
                            else:
                                error_msg = "Document embeddings are not loaded. Please load embeddings to answer document-related questions."
                            st.error(error_msg)
                    
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        # Capture and display the full traceback
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                        with st.expander("Error Details"):
                            st.code(error_details)
    
            # Display chat history
            for i, (role, message) in enumerate(reversed(st.session_state.chat_history)):
                if i >= 10:  # Only show the last 10 messages to avoid clutter
                    break
                if role == "user":
                    st.write(f"👤 **You:** {message}")
                else:
                    st.write(f"🤖 **Assistant:** {message}")
        else:
            st.info("Please load data using the sidebar to start chatting!")
    
    with col2:
        # Information panel
        st.subheader("Additional Information")
        
        # Show different information based on the last query mode
        if st.session_state.query_mode == "rag" and hasattr(st.session_state, 'last_sources'):
            with st.expander("Source Documents", expanded=True):
                for i, doc in enumerate(st.session_state.last_sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(f"```\n{doc.page_content}\n```")
                    st.markdown("---")
        
        elif st.session_state.query_mode == "stock" and st.session_state.stock_handler:
            # Show stock visualization options
            st.subheader("Stock Visualization")
            
            # Get all tickers
            all_tickers = st.session_state.stock_handler.tickers
            
            # Extract ticker from last query if available
            last_query = st.session_state.chat_history[-2][1] if len(st.session_state.chat_history) >= 2 else ""
            from financial_assistant.stock_analysis.analyzer import extract_ticker_from_query
            default_ticker = extract_ticker_from_query(last_query) or (all_tickers[0] if all_tickers else "")
            
            # Ticker selection
            ticker = st.selectbox("Select Ticker", all_tickers, index=all_tickers.index(default_ticker) if default_ticker in all_tickers else 0)
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            with col2:
                end_date = st.date_input("End Date", value=pd.to_datetime("2022-12-31"))
            
            # Chart type
            chart_type = st.selectbox("Chart Type", ["Price History", "Technical Indicators"])
            
            # Generate and display chart
            if st.button("Generate Chart"):
                with st.spinner("Generating chart..."):
                    if chart_type == "Price History":
                        chart_buf = st.session_state.stock_handler.plot_price_history(
                            ticker, 
                            start_date.strftime('%Y-%m-%d'), 
                            end_date.strftime('%Y-%m-%d')
                        )
                        if chart_buf:
                            st.image(Image.open(chart_buf), caption=f"{ticker} Price History")
                        else:
                            st.warning(f"No price data available for {ticker} in the selected date range.")
                    
                    elif chart_type == "Technical Indicators":
                        # Get technical indicators
                        indicators = st.session_state.stock_handler.get_technical_indicators(
                            ticker, 
                            start_date.strftime('%Y-%m-%d'), 
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if indicators:
                            st.write(f"Technical indicators for {ticker} as of {indicators['date'].strftime('%Y-%m-%d')}:")
                            st.write(f"RSI: {indicators['rsi']:.2f}")
                            st.write(f"MACD: {indicators['macd']:.2f}")
                            st.write(f"Bollinger Bands: Low=${indicators['bb_low']:.2f}, Mid=${indicators['bb_mid']:.2f}, High=${indicators['bb_high']:.2f}")
                            st.write(f"Volatility: {indicators['volatility']:.4f}")
                            st.write(f"ATR: {indicators['atr']:.4f}")
                        else:
                            st.warning(f"No technical indicator data available for {ticker} in the selected date range.")


def debug_section():
    """Add a debug section to the application."""
    with st.expander("Advanced Settings & Debug"):
        st.write("If you're experiencing issues, check the following:")
        st.write("1. Make sure Ollama is running (`ollama serve` in terminal)")
        st.write("2. Verify the model you selected is available in Ollama (`ollama list`)")
        st.write("3. Check that the embeddings directory contains valid Chroma database files")
        
        if st.button("Check Ollama Status"):
            ollama_running, models = check_ollama_server()
            if ollama_running:
                st.success(f"✅ Ollama is running with {len(models)} models available")
                st.write("Available models:", ", ".join(models))
            else:
                st.error("❌ Ollama server is not running or not responding")


def main():
    """Main function to run the application."""
    # Initialize session state
    init_session_state()
    
    # Set up the page
    ollama_running, available_models = setup_page()
    
    # Configure the sidebar
    data_source, embeddings_dir, stock_data_path, model_name, temperature, load_button = sidebar_config(available_models)
    
    # Load data if button is clicked
    if load_button:
        with st.spinner("Loading data..."):
            load_data(data_source, embeddings_dir, stock_data_path, model_name, temperature)
    
    # Create the chat interface
    chat_interface()
    
    # Add debug section
    debug_section()


if __name__ == "__main__":
    main() 