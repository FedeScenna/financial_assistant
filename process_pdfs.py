"""
PDF Embeddings Generator

This script creates embeddings from PDF files in the wikipedia folder using Chroma with GPU acceleration.
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import List, Dict
import PyPDF2
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch
import chromadb
from chromadb.config import Settings


def check_gpu_availability() -> bool:
    """
    Check if CUDA GPU is available for processing.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return True
    print("No GPU detected, using CPU")
    return False


def setup_gpu() -> None:
    """
    Configure GPU settings for optimal performance.
    """
    if torch.cuda.is_available():
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        # Set default tensor type to CUDA
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("GPU settings configured for optimal performance")


def read_pdf_file(file_path: str) -> str:
    """
    Read and extract text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return ""


def process_pdf_files(pdf_dir: str) -> List[Dict]:
    """
    Process all PDF files in the directory and extract text with metadata.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        
    Returns:
        List[Dict]: List of dictionaries containing text and metadata
    """
    documents = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Create progress bar for PDF processing
    with tqdm(pdf_files, desc="Processing PDF files", unit="file") as pbar:
        for pdf_file in pbar:
            # Update progress bar description with current file
            pbar.set_description(f"Processing {pdf_file.name}")
            
            text = read_pdf_file(str(pdf_file))
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_file.name,
                        "file_path": str(pdf_file)
                    }
                })
                # Update progress bar with success
                pbar.set_postfix({"status": "success"})
            else:
                # Update progress bar with failure
                pbar.set_postfix({"status": "failed"})
    
    return documents


def create_embeddings(pdf_dir: str, output_dir: str, model_name: str = "llama3.1:8b", use_gpu: bool = True):
    """
    Create embeddings from PDF files using Chroma with GPU acceleration.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        output_dir (str): Directory to save the embeddings
        model_name (str): Model to use for embeddings
        use_gpu (bool): Whether to use GPU for processing if available
    """
    print(f"Creating embeddings using {model_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check and setup GPU
    has_gpu = check_gpu_availability() if use_gpu else False
    if has_gpu:
        setup_gpu()
    
    # Process PDF files
    documents = process_pdf_files(pdf_dir)
    
    if not documents:
        print("No documents were successfully processed.")
        return
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Process all documents with progress bar
    all_chunks = []
    all_metadata = []
    
    print("\nCreating text chunks...")
    with tqdm(documents, desc="Chunking documents", unit="doc") as pbar:
        for doc in pbar:
            # Update progress bar description with current document
            pbar.set_description(f"Chunking {doc['metadata']['source']}")
            
            chunks = text_splitter.split_text(doc["text"])
            all_chunks.extend(chunks)
            # Add metadata to each chunk
            all_metadata.extend([doc["metadata"]] * len(chunks))
            
            # Update progress bar with chunk count
            pbar.set_postfix({"chunks": len(chunks)})
    
    print(f"\nCreated {len(all_chunks)} text chunks")
    
    # Create embeddings
    print("\nGenerating embeddings...")
    embeddings = OllamaEmbeddings(model=model_name)
    
    # Configure Chroma for GPU usage
    chroma_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=os.path.join(output_dir, "chroma_db"),
        allow_reset=True
    )
    
    # Create Chroma vector store with GPU support
    print("Building vector store...")
    vector_store = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadata,
        persist_directory=os.path.join(output_dir, "chroma_db"),
        client_settings=chroma_settings
    )
    
    # Persist the database
    print("Saving vector store...")
    vector_store.persist()
    
    # Save metadata about the embeddings
    metadata = {
        "model": model_name,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "num_chunks": len(all_chunks),
        "num_documents": len(documents),
        "gpu_used": has_gpu,
        "gpu_device": torch.cuda.get_device_name(0) if has_gpu else "cpu"
    }
    
    print("\nSaving metadata...")
    with open(os.path.join(output_dir, "index_metadata.pickle"), "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\nEmbeddings saved to {os.path.join(output_dir, 'chroma_db')}")


def main():
    parser = argparse.ArgumentParser(description="Create embeddings from PDF files")
    parser.add_argument("--pdf_dir", default="data/wikipedia", help="Directory containing PDF files")
    parser.add_argument("--output_dir", default="embeddings", help="Directory to save embeddings")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use for embeddings")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage even if available")
    args = parser.parse_args()
    
    create_embeddings(args.pdf_dir, args.output_dir, args.model, use_gpu=not args.no_gpu)


if __name__ == "__main__":
    main() 