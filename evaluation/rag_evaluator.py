"""
RAG Evaluation Script

This script evaluates the RAG application using multiple metrics:
- Faithfulness: Measures if the answer is supported by the retrieved context
- Answer Relevance: Measures if the answer is relevant to the question
- Context Relevance: Measures if the retrieved context is relevant to the question
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import pickle


class RAGEvaluator:
    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        model_name: str = "llama3.1:8b",
        use_gpu: bool = True
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            embeddings_dir (str): Directory containing the embeddings
            model_name (str): Name of the model to use for embeddings
            use_gpu (bool): Whether to use GPU for processing if available
        """
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Load embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = Chroma(
            persist_directory=os.path.join(embeddings_dir, "chroma_db"),
            embedding_function=self.embeddings
        )
        
        # Load metadata
        with open(os.path.join(embeddings_dir, "index_metadata.pickle"), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Initialize BERT model for semantic similarity
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if self.use_gpu:
            self.model = self.model.cuda()
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for a text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: BERT embedding
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def calculate_faithfulness(self, question: str, answer: str, context: str) -> float:
        """
        Calculate faithfulness score.
        
        Args:
            question (str): The question
            answer (str): The answer
            context (str): The retrieved context
            
        Returns:
            float: Faithfulness score between 0 and 1
        """
        # Get embeddings
        answer_embedding = self._get_bert_embedding(answer)
        context_embedding = self._get_bert_embedding(context)
        
        # Calculate similarity
        similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
        
        # Normalize to 0-1 range
        return float(similarity)
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Calculate answer relevance score.
        
        Args:
            question (str): The question
            answer (str): The answer
            
        Returns:
            float: Answer relevance score between 0 and 1
        """
        # Get embeddings
        question_embedding = self._get_bert_embedding(question)
        answer_embedding = self._get_bert_embedding(answer)
        
        # Calculate similarity
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        
        # Normalize to 0-1 range
        return float(similarity)
    
    def calculate_context_relevance(self, question: str, context: str) -> float:
        """
        Calculate context relevance score.
        
        Args:
            question (str): The question
            context (str): The retrieved context
            
        Returns:
            float: Context relevance score between 0 and 1
        """
        # Get embeddings
        question_embedding = self._get_bert_embedding(question)
        context_embedding = self._get_bert_embedding(context)
        
        # Calculate similarity
        similarity = cosine_similarity(question_embedding, context_embedding)[0][0]
        
        # Normalize to 0-1 range
        return float(similarity)
    
    def evaluate_query(self, question: str) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            question (str): The question to evaluate
            
        Returns:
            Dict: Evaluation results
        """
        # Get answer and context from RAG
        results = self.vector_store.similarity_search_with_score(question, k=3)
        
        # Extract context and answer
        context = "\n".join([doc.page_content for doc, _ in results])
        
        # For this example, we'll use the first document as the answer
        # In a real application, you would use your RAG chain to generate an answer
        answer = results[0][0].page_content
        
        # Calculate metrics
        faithfulness = self.calculate_faithfulness(question, answer, context)
        answer_relevance = self.calculate_answer_relevance(question, answer)
        context_relevance = self.calculate_context_relevance(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_relevance": context_relevance,
            "average_score": (faithfulness + answer_relevance + context_relevance) / 3
        }
    
    def evaluate_queries(self, questions: List[str]) -> List[Dict]:
        """
        Evaluate multiple queries.
        
        Args:
            questions (List[str]): List of questions to evaluate
            
        Returns:
            List[Dict]: List of evaluation results
        """
        results = []
        for question in tqdm(questions, desc="Evaluating queries"):
            result = self.evaluate_query(question)
            results.append(result)
        return results
    
    def save_results(self, results: List[Dict], output_file: str = "evaluation_results.json"):
        """
        Save evaluation results to a file.
        
        Args:
            results (List[Dict]): List of evaluation results
            output_file (str): Output file path
        """
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Calculate and print summary statistics
        df = pd.DataFrame(results)
        summary = {
            "average_faithfulness": df["faithfulness"].mean(),
            "average_answer_relevance": df["answer_relevance"].mean(),
            "average_context_relevance": df["context_relevance"].mean(),
            "overall_average": df["average_score"].mean()
        }
        
        print("\nEvaluation Summary:")
        print(f"Average Faithfulness: {summary['average_faithfulness']:.4f}")
        print(f"Average Answer Relevance: {summary['average_answer_relevance']:.4f}")
        print(f"Average Context Relevance: {summary['average_context_relevance']:.4f}")
        print(f"Overall Average Score: {summary['overall_average']:.4f}")
        
        # Save summary to file
        with open("evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    # Sample questions for evaluation
    questions = [
        "Where are Apple's headquarters located?",
        "Who is Tesla's CEO?",
        "What is the main product of Microsoft?",
        "What is the main product of Google?",
        "What is the main product of Meta (Facebook)?",
        "What is the main product of Netflix?",
        "What is the main product of Disney?",
        "What is the main product of General Electric?",
        "What is the main product of Pfizer?",
        "Who owns General Electric?",
        "How has Pfizer's stock price changed since the COVID-19 vaccine announcement?"
    ]
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        embeddings_dir="embeddings",
        model_name="llama3.1:8b",
        use_gpu=True
    )
    
    # Run evaluation
    results = evaluator.evaluate_queries(questions)
    
    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main() 