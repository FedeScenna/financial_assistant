"""
Test script to verify all imports are working correctly.
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from src.utils.helpers import check_ollama_server, check_gpu, is_stock_query
    from src.rag.engine import get_conversation_chain
    from src.stock_analysis.analyzer import load_stock_data, process_stock_query
    from src.app import main
    
    print("✅ All imports successful!")
    print("\nTesting individual components:")
    
    # Test helper functions
    print("\nTesting helper functions...")
    print(f"Ollama server status: {check_ollama_server()}")
    print(f"GPU status: {check_gpu()}")
    print(f"Stock query test: {is_stock_query('What is the stock price of AAPL?')}")
    
    # Test RAG engine
    print("\nTesting RAG engine...")
    try:
        chain = get_conversation_chain()
        print("✅ RAG engine initialized successfully")
    except Exception as e:
        print(f"❌ RAG engine initialization failed: {str(e)}")
    
    # Test stock analysis
    print("\nTesting stock analysis...")
    try:
        data = load_stock_data()
        print("✅ Stock data loaded successfully")
    except Exception as e:
        print(f"❌ Stock data loading failed: {str(e)}")
    
except ImportError as e:
    print(f"❌ Import failed: {str(e)}")
    print("\nPlease check that:")
    print("1. The src directory is in the correct location")
    print("2. All required packages are installed")
    print("3. The Python path includes the project root directory") 