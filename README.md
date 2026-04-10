# RAG-eval
Project for evaluating the RAG pipelines

## Setup
install required packages:
`pip install pandas langchain langchain-openai langgraph numpy sentence-transformers langchain-huggingface`

## Usage
1. Run retriever evaluation:
`python retrieval.py`
2. Run generator evaluation:
`python generator.py`

## Used models 
- retriever: `BAAI/bge-m3`
- reranker: `BAAI/bge-reranker-v2-m3`
- generator: `gpt-4o-mini`
- embedding: `all-MiniLM-L6-v2`

### Details
- The retriever and reranker are evaluated with DCG@10 (10 most relevant documents are considered).
- The generator is evaluated semantic similarity between the generated answer and the reference answer using cosine similarity of their embeddings.
- Additionally the generated answer and its documents are evaluated with AIS (Attributable to Identified Sources) metric. In this case the generator model is used as a judge to evaluate the quality of the generated answer and its attribution to the retrieved documents.
