# Financial Assistant with RAG and Stock Analysis

A comprehensive financial assistant application that combines RAG (Retrieval-Augmented Generation) with stock market analysis capabilities. The application uses local LLMs via Ollama and provides both document-based and stock market insights.

## Features

- **Document Analysis**: Process and query financial documents using RAG
- **Stock Market Analysis**: Analyze stock prices and technical indicators
- **Local LLM Support**: Uses Ollama for local inference
- **GPU Acceleration**: Supports GPU acceleration for faster processing
- **Interactive UI**: Built with Streamlit for a user-friendly interface

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- Ollama installed and running locally

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-assistant.git
cd financial-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Ollama server:
```bash
ollama serve
```

## Usage

1. Process PDF documents:
```bash
python process_pdfs.py --pdf_dir data/wikipedia --output_dir embeddings
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure

```
financial_assistant/
├── app.py                 # Main Streamlit application
├── process_pdfs.py        # PDF processing and embedding generation
├── rag/                   # RAG implementation
├── stock_analysis/        # Stock market analysis
├── utils/                 # Utility functions
├── data/                  # Data directory (not included in repo)
└── embeddings/           # Generated embeddings (not included in repo)
```

## Configuration

The application can be configured through the Streamlit interface:
- Select data sources (Document Embeddings, Stock Price Data, or Both)
- Choose the Ollama model
- Adjust temperature for response generation
- Configure visualization options for stock data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- UI created with [Streamlit](https://streamlit.io/) 