"""
RAGAS Evaluation Script

This script evaluates the RAG application using RAGAS metrics.
"""

import os
import torch
import pandas as pd
from datetime import datetime
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

# Test cases
questions = [
    "What does Adobe specialize in?",
    "Who is the CEO of JPMorgan Chase?",
    "When was Netflix founded?",
    "How do the business models of Amazon and Walmart differ?",
    "Compare the founding histories of Apple and Microsoft.",
    "Tell me about Salesforce.",
    "What is Visa known for?",
    "What major acquisitions has Microsoft made in the past 10 years?",
    "How has Tesla evolved since its IPO?",
    "Why is Berkshire Hathaway unique among S&P 500 companies?",
    "What regulatory challenges has Meta faced in recent years?",
    "What is the market share of a fictional company like 'ZyborTech'?",
    "How does S&P500 member OpenAI operate?",
    "Where is Meta Platforms headquartered?",
    "Where is Johnson & Johnson headquartered?",
    "Where is Goldman Sachs headquartered?",
    "When was Nvidia founded?",
    "When was Coca-Cola founded?",
    "When was Amazon founded?"
]

expected_answers = [
    "Adobe specializes in software for creative professionals, including Photoshop, Illustrator, and Acrobat.",
    "Jamie Dimon is the CEO of JPMorgan Chase.",
    "Netflix was founded in 1997.",
    "Amazon primarily operates as an online marketplace and cloud provider, while Walmart focuses on physical retail locations with growing e-commerce presence.",
    "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
    "Salesforce is a cloud-based software company known for its customer relationship management (CRM) platform.",
    "Visa is known for its global electronic payment network and credit card services.",
    "Major acquisitions include LinkedIn, GitHub, and Activision Blizzard.",
    "Tesla has expanded from electric cars to energy products, significantly increased production capacity, and grown into one of the largest auto manufacturers by market value.",
    "Berkshire Hathaway is a conglomerate holding company led by Warren Buffett, notable for its diverse investments and decentralized management.",
    "Meta has faced antitrust investigations, privacy concerns, and content moderation scrutiny in the US and EU.",
    "There is no available information on a company named ZyborTech in the S&P 500.",
    "OpenAI is not a member of the S&P 500.",
    "Meta Platforms is headquartered in Menlo Park, California.",
    "Johnson & Johnson is headquartered in New Brunswick, New Jersey.",
    "Goldman Sachs is headquartered in New York City, New York.",
    "Nvidia was founded in 1993.",
    "Coca-Cola was founded in 1892.",
    "Amazon was founded in 1994."
]

def get_financial_prompt_template():
    """
    Returns a prompt template for financial analysis.
    """
    template = """You are a helpful financial assistant. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_conversation_chain(vector_store, model_name="llama3.1:8b", temperature=0.7):
    """
    Create a conversation chain for RAG.
    """
    try:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Get the financial prompt template
        PROMPT = get_financial_prompt_template()

        # Configure Ollama to use GPU if available
        if torch.cuda.is_available():
            print("Using GPU for model inference")
            llm = Ollama(
                model=model_name,
                temperature=temperature,
                callback_manager=callback_manager,
                num_gpu=1
            )
        else:
            print("Using CPU for model inference")
            llm = Ollama(
                model=model_name,
                temperature=temperature,
                callback_manager=callback_manager
            )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True
        )
        return conversation_chain
    except Exception as e:
        print(f"Error in get_conversation_chain: {str(e)}")
        raise

def evaluate_rag():
    """
    Evaluate the RAG system using RAGAS metrics.
    """
    try:
        # Configure GPU usage
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["CHROMA_GPU"] = "1"
            print("✅ GPU Available")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["CHROMA_GPU"] = "0"
            print("⚠️ No GPU detected. Running on CPU.")

        # Initialize embeddings and vector store
        embeddings = OllamaEmbeddings(model="llama3.1:8b")
        vector_store = Chroma(
            persist_directory="embeddings",
            embedding_function=embeddings
        )

        # Initialize conversation chain
        conversation_chain = get_conversation_chain(vector_store)

        # Prepare evaluation data
        evaluation_data = []
        for question, expected_answer in zip(questions, expected_answers):
            print(f"\nProcessing question: {question}")
            # Get response from RAG system
            response = conversation_chain({"question": question})
            answer = response["answer"]
            contexts = [doc.page_content for doc in response["source_documents"]]

            evaluation_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": expected_answer
            })

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(evaluation_data)

        # Define metrics
        metrics = [
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            answer_correctness
        ]

        # Run evaluation
        print("\nStarting RAGAS evaluation...")
        result = evaluate(
            dataset,
            metrics=metrics
        )

        # Create detailed results DataFrame
        detailed_results = []
        for idx, (question, expected_answer) in enumerate(zip(questions, expected_answers)):
            detailed_results.append({
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": evaluation_data[idx]["answer"],
                "answer_relevancy": result["answer_relevancy"][idx],
                "faithfulness": result["faithfulness"][idx],
                "context_recall": result["context_recall"][idx],
                "context_precision": result["context_precision"][idx],
                "answer_correctness": result["answer_correctness"][idx],
                "contexts": evaluation_data[idx]["contexts"]
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(detailed_results)

        # Add summary statistics
        summary_stats = {
            "Metric": ["answer_relevancy", "faithfulness", "context_recall", "context_precision", "answer_correctness"],
            "Mean": [
                result["answer_relevancy"].mean(),
                result["faithfulness"].mean(),
                result["context_recall"].mean(),
                result["context_precision"].mean(),
                result["answer_correctness"].mean()
            ],
            "Std": [
                result["answer_relevancy"].std(),
                result["faithfulness"].std(),
                result["context_recall"].std(),
                result["context_precision"].std(),
                result["answer_correctness"].std()
            ]
        }
        summary_df = pd.DataFrame(summary_stats)

        # Save results to CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_filename = f"ragas_evaluation_results_{timestamp}.csv"
        summary_filename = f"ragas_evaluation_summary_{timestamp}.csv"
        
        results_df.to_csv(detailed_filename, index=False)
        summary_df.to_csv(summary_filename, index=False)

        # Print results
        print("\nEvaluation Results:")
        print("===================")
        print("\nDetailed Results:")
        print(results_df)
        print("\nSummary Statistics:")
        print(summary_df)
        print(f"\nResults saved to {detailed_filename} and {summary_filename}")

        return results_df, summary_df

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_rag() 