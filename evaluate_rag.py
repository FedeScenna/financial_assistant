"""
RAG Evaluation Script

This script evaluates the performance of the RAG application based on the obtained embeddings.
It measures metrics like retrieval accuracy, answer relevance, and response quality.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
import spacy
from typing import List, Tuple

from rag.engine import load_vector_store, get_conversation_chain
from utils.helpers import check_gpu


class RAGEvaluator:
    """Class to evaluate the RAG application performance."""
    
    def __init__(self, embeddings_dir, model_name="llama3.1:8b", temperature=0.5, evaluation_data=None):
        """
        Initialize the RAG evaluator.
        
        Args:
            embeddings_dir (str): Directory containing the embeddings
            model_name (str): Name of the model to use for generation
            temperature (float): Temperature parameter for generation
            evaluation_data (str): Path to the evaluation data file
        """
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name
        self.temperature = temperature
        self.evaluation_data = evaluation_data
        self.vector_store = None
        self.conversation_chain = None
        self.evaluation_results = []
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Check GPU availability
        self.has_gpu, self.gpu_info = check_gpu()
        print(f"GPU Status: {'Available - ' + self.gpu_info if self.has_gpu else 'Not available'}")
        
    def load_model(self):
        """Load the RAG model and vector store."""
        print(f"Loading vector store from {self.embeddings_dir}")
        try:
            self.vector_store = load_vector_store(self.embeddings_dir, self.model_name)
            self.conversation_chain = get_conversation_chain(
                self.vector_store, 
                self.model_name, 
                self.temperature
            )
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
            
    def load_evaluation_data(self, evaluation_data_path=None):
        """
        Load evaluation data from a JSON file.
        
        The JSON file should contain a list of objects with the following structure:
        [
            {
                "query": "What was Apple's revenue in 2022?",
                "expected_answer": "Apple's revenue in 2022 was $394.3 billion.",
                "relevant_documents": ["Apple_Annual_Report_2022.pdf"]
            },
            ...
        ]
        
        Args:
            evaluation_data_path (str): Path to the evaluation data file
            
        Returns:
            list: List of evaluation data objects
        """
        data_path = evaluation_data_path or self.evaluation_data
        
        if not data_path or not os.path.exists(data_path):
            print("No evaluation data provided or file doesn't exist.")
            print("Creating a default evaluation dataset...")
            
            # Create default evaluation data with some common financial questions
            default_data = [
                {
                    "query": "Where are AES headquarters located?",
                    "expected_keywords": ["Virginia","USA","Arlington"]
                },
                {
                    "query": "Which are the main activities of ACN?",
                    "expected_keywords": ["products", "services", "sales", "customers", "business"]
                },
                {
                    "query": "What is the difference between stocks and bonds?",
                    "expected_keywords": ["equity", "debt", "ownership", "interest", "risk"]
                },
                {
                    "query": "What are financial statements?",
                    "expected_keywords": ["balance sheet", "income statement", "cash flow", "reporting", "accounting"]
                },
                {
                    "query": "How do investors analyze stocks?",
                    "expected_keywords": ["valuation", "metrics", "ratios", "analysis", "fundamental"]
                }
            ]
            
            return default_data
        
        try:
            with open(data_path, 'r') as f:
                evaluation_data = json.load(f)
            print(f"Loaded {len(evaluation_data)} evaluation queries from {data_path}")
            return evaluation_data
        except Exception as e:
            print(f"Error loading evaluation data: {str(e)}")
            return []
    
    def evaluate_retrieval(self, query, top_k=3):
        """
        Evaluate the retrieval component of the RAG system.
        
        Args:
            query (str): The query to evaluate
            top_k (int): Number of documents to retrieve
            
        Returns:
            list: Retrieved documents
        """
        if not self.vector_store:
            print("Vector store not loaded. Call load_model() first.")
            return []
        
        # Get the retrieved documents
        retrieved_docs = self.vector_store.similarity_search(query, k=top_k)
        return retrieved_docs
    
    def evaluate_answer(self, query):
        """
        Evaluate the answer generation component of the RAG system.
        
        Args:
            query (str): The query to evaluate
            
        Returns:
            tuple: (answer, source_documents)
        """
        if not self.conversation_chain:
            print("Conversation chain not loaded. Call load_model() first.")
            return None, []
        
        # Get the generated answer
        result = self.conversation_chain({"question": query})
        return result['answer'], result.get('source_documents', [])
    
    def calculate_retrieval_precision(self, retrieved_docs, relevant_docs=None):
        """
        Calculate the precision of document retrieval.
        
        Args:
            retrieved_docs (list): List of retrieved documents
            relevant_docs (list): List of relevant document names or None
            
        Returns:
            float: Precision score (0-1)
        """
        if not relevant_docs:
            return None
        
        # Extract retrieved document names
        retrieved_doc_names = [os.path.basename(doc.metadata.get('source', '')) for doc in retrieved_docs]
        
        # Calculate precision
        hits = sum(1 for doc in retrieved_doc_names if any(rel_doc in doc for rel_doc in relevant_docs))
        precision = hits / len(retrieved_docs) if retrieved_docs else 0
        
        return precision
    
    def calculate_keyword_coverage(self, answer, expected_keywords):
        """
        Calculate the keyword coverage in the generated answer using spaCy.
        
        Args:
            answer (str): Generated answer
            expected_keywords (list): List of expected keywords
            
        Returns:
            float: Keyword coverage score (0-1)
        """
        if not expected_keywords:
            return None
        
        # Process the answer with spaCy
        doc = self.nlp(answer.lower())
        
        # Get lemmatized tokens and named entities
        answer_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        answer_entities = [ent.text.lower() for ent in doc.ents]
        
        # Combine tokens and entities
        answer_terms = set(answer_tokens + answer_entities)
        
        # Calculate coverage
        hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_terms)
        coverage = hits / len(expected_keywords) if expected_keywords else 0
        
        return coverage
    
    def calculate_bleu_score(self, answer, expected_answer):
        """
        Calculate BLEU score for answer quality using spaCy tokenization.
        
        Args:
            answer (str): Generated answer
            expected_answer (str): Expected answer
            
        Returns:
            float: BLEU score (0-1)
        """
        if not expected_answer:
            return None
        
        # Process both answers with spaCy
        answer_doc = self.nlp(answer.lower())
        expected_doc = self.nlp(expected_answer.lower())
        
        # Get tokens (excluding stop words and punctuation)
        answer_tokens = [token.text for token in answer_doc if not token.is_stop and not token.is_punct]
        expected_tokens = [token.text for token in expected_doc if not token.is_stop and not token.is_punct]
        
        # Calculate BLEU score
        try:
            bleu_score = sentence_bleu([expected_tokens], answer_tokens)
            return bleu_score
        except Exception as e:
            print(f"Error calculating BLEU score: {str(e)}")
            return 0
    
    def evaluate(self):
        """Run the full evaluation and return results."""
        # Load the evaluation data
        evaluation_data = self.load_evaluation_data()
        
        if not evaluation_data:
            print("No evaluation data available.")
            return []
        
        if not self.load_model():
            print("Failed to load model. Aborting evaluation.")
            return []
        
        results = []
        
        # Evaluate each query
        for item in tqdm(evaluation_data, desc="Evaluating queries"):
            query = item["query"]
            expected_answer = item.get("expected_answer")
            expected_keywords = item.get("expected_keywords", [])
            relevant_docs = item.get("relevant_documents", [])
            
            # Measure retrieval time and performance
            retrieved_docs = self.evaluate_retrieval(query)
            
            # Measure answer generation time and quality
            answer, source_docs = self.evaluate_answer(query)
            
            # Calculate metrics
            retrieval_precision = self.calculate_retrieval_precision(retrieved_docs, relevant_docs)
            keyword_coverage = self.calculate_keyword_coverage(answer, expected_keywords)
            bleu_score = self.calculate_bleu_score(answer, expected_answer)
            
            # Store results
            result = {
                "query": query,
                "answer": answer,
                "retrieved_docs": [doc.metadata.get('source', '') for doc in retrieved_docs],
                "source_docs": [doc.metadata.get('source', '') for doc in source_docs],
                "metrics": {
                    "retrieval_precision": retrieval_precision,
                    "keyword_coverage": keyword_coverage,
                    "bleu_score": bleu_score
                }
            }
            
            results.append(result)
        
        self.evaluation_results = results
        return results
    
    def generate_report(self, output_file=None):
        """
        Generate an evaluation report.
        
        Args:
            output_file (str): Path to the output file for the report
            
        Returns:
            dict: Aggregated metrics
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate() first.")
            return {}
        
        # Aggregate metrics
        metrics = {
            "retrieval_precision": [],
            "keyword_coverage": [],
            "bleu_score": []
        }
        
        for result in self.evaluation_results:
            result_metrics = result["metrics"]
            for key in metrics.keys():
                if result_metrics.get(key) is not None:
                    metrics[key].append(result_metrics[key])
        
        # Calculate averages
        aggregated_metrics = {}
        for key, values in metrics.items():
            if values:
                aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
                aggregated_metrics[f"min_{key}"] = min(values)
                aggregated_metrics[f"max_{key}"] = max(values)
            else:
                aggregated_metrics[f"avg_{key}"] = None
                aggregated_metrics[f"min_{key}"] = None
                aggregated_metrics[f"max_{key}"] = None
        
        # Print report
        print("\n===== RAG EVALUATION REPORT =====")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"Number of queries evaluated: {len(self.evaluation_results)}")
        print("\nMetrics:")
        for key, value in aggregated_metrics.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
        
        # Save detailed results if output file is provided
        if output_file:
            results_dict = {
                "metadata": {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "embeddings_dir": self.embeddings_dir,
                    "num_queries": len(self.evaluation_results)
                },
                "aggregated_metrics": aggregated_metrics,
                "detailed_results": self.evaluation_results
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\nDetailed results saved to {output_file}")
        
        return aggregated_metrics
    
    def visualize_results(self, output_dir=None):
        """
        Visualize the evaluation results.
        
        Args:
            output_dir (str): Directory to save the visualizations
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate() first.")
            return
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics for visualization
        metrics = {
            "retrieval_precision": [],
            "keyword_coverage": [],
            "bleu_score": []
        }
        
        queries = []
        
        for result in self.evaluation_results:
            queries.append(result["query"])
            result_metrics = result["metrics"]
            for key in metrics.keys():
                if result_metrics.get(key) is not None:
                    metrics[key].append(result_metrics[key])
                else:
                    metrics[key].append(0)  # Replace None with 0 for visualization
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            "Query": queries,
            "Retrieval Precision": metrics["retrieval_precision"],
            "Keyword Coverage": metrics["keyword_coverage"],
            "BLEU Score": metrics["bleu_score"]
        })
        
        # Plot metrics for each query
        plt.figure(figsize=(12, 6))
        df_melted = pd.melt(df, id_vars=["Query"], var_name="Metric", value_name="Score")
        sns.barplot(x="Query", y="Score", hue="Metric", data=df_melted)
        plt.title("RAG Evaluation Metrics by Query")
        plt.xlabel("Query")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "rag_metrics_by_query.png"))
        else:
            plt.show()
        
        # Plot average metrics
        plt.figure(figsize=(8, 6))
        avg_metrics = df.iloc[:, 1:].mean()
        sns.barplot(x=avg_metrics.index, y=avg_metrics.values)
        plt.title("Average RAG Evaluation Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Average Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "rag_avg_metrics.png"))
        else:
            plt.show()


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate RAG application performance")
    parser.add_argument("--embeddings_dir", required=True, help="Directory containing the embeddings")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter for generation")
    parser.add_argument("--eval_data", help="Path to evaluation data JSON file")
    parser.add_argument("--output_file", help="Path to output file for the evaluation report")
    parser.add_argument("--visualize", action="store_true", help="Visualize the evaluation results")
    parser.add_argument("--output_dir", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Create the evaluator
    evaluator = RAGEvaluator(
        args.embeddings_dir,
        args.model,
        args.temperature,
        args.eval_data
    )
    
    # Run the evaluation
    evaluator.evaluate()
    
    # Generate the report
    evaluator.generate_report(args.output_file)
    
    # Visualize the results if requested
    if args.visualize:
        evaluator.visualize_results(args.output_dir)


if __name__ == "__main__":
    main() 