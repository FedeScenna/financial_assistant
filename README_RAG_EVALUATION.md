# RAG Evaluation Tools

This directory contains tools for evaluating the performance of the Retrieval-Augmented Generation (RAG) components in the Financial Assistant application. These tools allow you to assess how well the RAG system is retrieving relevant documents and generating accurate answers based on the embeddings.

## Overview

The evaluation toolkit includes:

1. `evaluate_rag.py` - The main evaluation script that tests RAG performance
2. `create_sample_embeddings.py` - A utility to generate sample embeddings for testing
3. `sample_evaluation_data.json` - A set of test queries with expected answers and keywords

## Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install langchain langchain_ollama langchain_community faiss-cpu numpy pandas matplotlib seaborn sklearn nltk tqdm torch
```

You must have Ollama installed and running with the appropriate model available (default: llama3.1:8b).

## Using the Evaluation Tools

### Step 1: Create Sample Embeddings (Optional)

If you don't have embeddings to test with, you can create sample embeddings using:

```bash
python create_sample_embeddings.py --output_dir sample_embeddings --model llama3.1:8b
```

This creates a directory of sample embeddings based on financial texts for testing purposes.

### Step 2: Run the Evaluation

To evaluate your RAG system:

```bash
python evaluate_rag.py --embeddings_dir your_embeddings_dir --model llama3.1:8b --eval_data sample_evaluation_data.json --output_file evaluation_results.json --visualize --output_dir evaluation_output
```

Parameters:
- `--embeddings_dir`: Directory containing your embeddings (required)
- `--model`: Model to use for generation (default: llama3.1:8b)
- `--temperature`: Temperature parameter for generation (default: 0.7)
- `--eval_data`: Path to evaluation data JSON file (optional)
- `--output_file`: Path to save the evaluation report (optional)
- `--visualize`: Generate visualizations of the results
- `--output_dir`: Directory to save visualizations

If you don't provide an evaluation data file, the script will use a default set of financial questions.

### Step 3: Analyze the Results

The evaluation reports several metrics:

1. **Retrieval Precision**: Measures how well the system retrieves relevant documents
2. **Keyword Coverage**: Evaluates whether the generated answers contain expected keywords/topics
3. **BLEU Score**: Compares the generated answers with expected answers for similarity

The script produces:
- A console output with summary metrics
- A detailed JSON report (if output_file is specified)
- Visualization graphs (if --visualize is specified)

## Creating Custom Evaluation Data

You can create your own evaluation data JSON file with the following structure:

```json
[
  {
    "query": "Your test question here?",
    "expected_answer": "The ideal answer to the question",
    "expected_keywords": ["key", "terms", "that", "should", "appear"],
    "relevant_documents": ["doc1.pdf", "doc2.pdf"]
  },
  ...
]
```

The "expected_answer" and "relevant_documents" fields are optional but recommended for comprehensive evaluation.

## Interpreting the Results

- **Retrieval Precision**: Values closer to 1.0 indicate the system is retrieving the correct documents
- **Keyword Coverage**: Values closer to 1.0 mean the answers contain more of the expected key concepts
- **BLEU Score**: Higher values (closer to 1.0) indicate better textual similarity to expected answers

## Troubleshooting

Common issues:
1. **Dimension mismatch errors**: Ensure you're using the same embedding model that created the embeddings
2. **Ollama connection errors**: Verify Ollama is running with `ollama serve`
3. **Model not found**: Make sure you have the specified model pulled in Ollama with `ollama pull llama3.1:8b`

## Further Improvements

Potential enhancements to the evaluation system:
- Add query latency measurements
- Implement more sophisticated metrics like ROUGE or BERTScore
- Create a web UI for interactive evaluation
- Add support for few-shot learning tests 