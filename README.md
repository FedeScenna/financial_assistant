# Financial Assistant with RAG and Stock Analysis

A comprehensive financial analysis tool that combines Retrieval-Augmented Generation (RAG) with stock market analysis capabilities. The application processes financial documents, creates embeddings, and provides intelligent responses to financial queries.

## 🚀 Features

- **Document Processing**: Efficient PDF processing with GPU support
- **Vector Storage**: Uses Milvus for scalable vector storage
- **RAG Implementation**: Advanced retrieval-augmented generation for accurate financial information
- **Stock Analysis**: Bayesian modeling for stock return predictions with GPU acceleration
- **Evaluation Tools**: Comprehensive metrics for RAG performance evaluation
- **GPU Acceleration**: Optional GPU support for faster processing

## 📁 Project Structure

```
financial_assistant/
├── src/                    # Main application code
│   ├── app.py             # Main Streamlit application
│   ├── utils/             # Utility functions
│   ├── rag/               # RAG implementation
│   ├── stock_analysis/    # Stock analysis tools
│   ├── document_processing/ # Document processing utilities
│   └── evaluation/        # RAG evaluation tools and metrics
├── data/                  # Data storage
│   ├── wikipedia/         # Financial documents
│   ├── statement_analysis/
│   └── income_statements/
├── embeddings/            # Vector embeddings storage
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/FedeScenna/financial_assistant.git
cd financial_assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Milvus (for vector storage):
```bash
# Using Docker
docker-compose up -d
```

4. Install Ollama (for LLM):
```bash
# Windows
curl -L https://ollama.ai/download/ollama-windows-amd64.zip -o ollama.zip
Expand-Archive ollama.zip
cd ollama
./ollama.exe serve

# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh
```

## 🚀 Usage

1. Start the application:
```bash
python src/app.py
```

2. Process PDF documents:
```bash
python src/document_processing/process_pdfs.py
```

3. Run RAG evaluations:
```bash
python src/evaluation/rag_evaluator.py
```

4. Run Bayesian stock predictions:
```bash
python src/stock_analysis/bayesian_model.py
```

## 🔧 Configuration

### GPU Support
The application automatically detects and uses GPU if available. To force CPU usage:
```bash
export USE_GPU=false
```

### Model Selection
Default model is "llama3.1:8b". To change:
```bash
export MODEL_NAME="your-preferred-model"
```

## 📊 Evaluation Metrics

The application includes comprehensive evaluation tools that measure:
- Retrieval accuracy
- Answer relevance
- Context relevance
- Response quality

Results are stored in `evaluation_results.json` and summarized in `evaluation_summary.json`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Troubleshooting

1. **Milvus Connection Issues**:
   - Ensure Milvus is running: `docker ps | grep milvus`
   - Check connection settings in `src/utils/helpers.py`

2. **Ollama Server Issues**:
   - Verify Ollama is running: `curl http://localhost:11434/api/version`
   - Check model availability: `ollama list`

3. **GPU Support**:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

4. **Bayesian Model Issues**:
   - Ensure PyMC is properly installed: `python -c "import pymc as pm; print(pm.__version__)"`
   - Check GPU memory: `nvidia-smi`
   - For memory issues, reduce the number of chains or samples in `bayesian_model.py`

## 📞 Support

For issues and feature requests, please open an issue in the GitHub repository. 