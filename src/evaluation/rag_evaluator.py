import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import ContextPrecision, ContextRecall

# Create a dataset with questions and their contexts
opik_client = opik.Opik()
dataset = opik_client.get_or_create_dataset("RAG evaluation dataset")
dataset.insert([
  {
    "question": "What does Adobe specialize in?",
    "expected_answer": "Adobe specializes in software for creative professionals, including Photoshop, Illustrator, and Acrobat.",
    "category": "basic_factual"
  },
  {
    "question": "Who is the CEO of JPMorgan Chase?",
    "expected_answer": "Jamie Dimon is the CEO of JPMorgan Chase.",
    "category": "basic_factual"
  },
  {
    "question": "When was Netflix founded?",
    "expected_answer": "Netflix was founded in 1997.",
    "category": "basic_factual"
  },
  {
    "question": "How do the business models of Amazon and Walmart differ?",
    "expected_answer": "Amazon primarily operates as an online marketplace and cloud provider, while Walmart focuses on physical retail locations with growing e-commerce presence.",
    "category": "comparative_reasoning"
  },
  {
    "question": "Compare the founding histories of Apple and Microsoft.",
    "expected_answer": "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
    "category": "comparative_reasoning"
  },
  {
    "question": "Tell me about Salesforce.",
    "expected_answer": "Salesforce is a cloud-based software company known for its customer relationship management (CRM) platform.",
    "category": "entity_disambiguation"
  },
  {
    "question": "What is Visa known for?",
    "expected_answer": "Visa is known for its global electronic payment network and credit card services.",
    "category": "entity_disambiguation"
  },
  {
    "question": "What major acquisitions has Microsoft made in the past 10 years?",
    "expected_answer": "Major acquisitions include LinkedIn, GitHub, and Activision Blizzard.",
    "category": "temporal_awareness"
  },
  {
    "question": "How has Tesla evolved since its IPO?",
    "expected_answer": "Tesla has expanded from electric cars to energy products, significantly increased production capacity, and grown into one of the largest auto manufacturers by market value.",
    "category": "temporal_awareness"
  },
  {
    "question": "Why is Berkshire Hathaway unique among S&P 500 companies?",
    "expected_answer": "Berkshire Hathaway is a conglomerate holding company led by Warren Buffett, notable for its diverse investments and decentralized management.",
    "category": "contextual_understanding"
  },
  {
    "question": "What regulatory challenges has Meta faced in recent years?",
    "expected_answer": "Meta has faced antitrust investigations, privacy concerns, and content moderation scrutiny in the US and EU.",
    "category": "contextual_understanding"
  },
  {
    "question": "What is the market share of a fictional company like 'ZyborTech'?",
    "expected_answer": "There is no available information on a company named ZyborTech in the S&P 500.",
    "category": "edge_case"
  },
  {
    "question": "How does S&P500 member OpenAI operate?",
    "expected_answer": "OpenAI is not a member of the S&P 500.",
    "category": "edge_case"
  }
])

def rag_task(item):
    # Simulate RAG pipeline
    output = "<LLM response placeholder>"

    return {
        "output": output
    }

# Run the evaluation
result = evaluate(
    dataset=dataset,
    task=rag_task,
    scoring_metrics=[
        ContextPrecision(),
        ContextRecall()
    ],
    experiment_name="rag_evaluation"
)

