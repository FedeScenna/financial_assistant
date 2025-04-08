# RAG Evaluation Script

This script evaluates the performance of the RAG application using three key metrics:
- Faithfulness: Measures if the answer is supported by the retrieved context
- Answer Relevance: Measures if the answer is relevant to the question
- Context Relevance: Measures if the retrieved context is relevant to the question

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended for faster processing)
3. Ollama installed and running with the llama3.1:8b model

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your embeddings are in the correct directory structure:
```
embeddings/
├── chroma_db/
└── index_metadata.pickle
```

2. Run the evaluation script:
```bash
python rag_evaluator.py
```

The script will:
- Load the existing embeddings from the Chroma database
- Evaluate a set of sample financial questions
- Calculate faithfulness, answer relevance, and context relevance scores
- Save detailed results to `evaluation_results.json`
- Save summary statistics to `evaluation_summary.json`
- Print summary statistics to the console

## Customization

You can modify the script to:
- Change the model used for embeddings by updating the `model_name` parameter
- Add your own evaluation questions by modifying the `questions` list in the `main()` function
- Adjust the number of retrieved documents by changing the `k` parameter in `evaluate_query()`
- Modify the evaluation metrics by updating the calculation methods

## Output Format

The evaluation results are saved in two files:

1. `evaluation_results.json`: Contains detailed results for each question:
```json
[
  {
    "question": "What is the current market cap of Apple?",
    "answer": "...",
    "context": "...",
    "faithfulness": 0.85,
    "answer_relevance": 0.92,
    "context_relevance": 0.88,
    "average_score": 0.88
  },
  ...
]
```

2. `evaluation_summary.json`: Contains summary statistics:
```json
{
  "average_faithfulness": 0.85,
  "average_answer_relevance": 0.88,
  "average_context_relevance": 0.82,
  "overall_average": 0.85
}
```

## Troubleshooting

1. If you encounter CUDA out-of-memory errors:
   - Reduce the batch size in the BERT model
   - Use a smaller model for embeddings
   - Process fewer questions at a time

2. If the embeddings fail to load:
   - Verify that the Chroma database exists in the correct location
   - Check that the metadata file is present and readable
   - Ensure Ollama is running with the correct model

3. If the scores seem unusually low:
   - Check the quality of your embeddings
   - Verify that the retrieved documents are relevant
   - Consider adjusting the similarity thresholds 