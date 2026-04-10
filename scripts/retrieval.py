import os
import ast
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from typing import Dict, List


def compute_ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at k."""
    # Compute DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)

    # Compute IDCG@k
    ideal_rels = sorted(qrels.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k]):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_pipeline(retriever, reranker, queries, corpus, qrels, top_k_retrieve=100, top_k_ndcg=10, results_path="results.csv"):
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())

    print("\nEncoding corpus for first-stage retrieval...")
    # Compute embeddings for all documents
    corpus_embeddings = retriever.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

    ndcg_scores = []
    results = []

    print("Running pipeline on queries...")
    for qid, query_text in tqdm(queries.items(), total=len(queries)):
        if qid not in qrels:
            continue

        # --- STAGE 1: First-Stage Retrieval (BGE-M3 Dense) ---
        query_embedding = retriever.encode(query_text, convert_to_tensor=True)

        # Calculate cosine similarity between query and all documents
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # Get top-K initial candidates
        top_results = torch.topk(cos_scores, k=min(top_k_retrieve, len(corpus_texts)))
        candidate_indices = top_results.indices.cpu().numpy()

        candidate_doc_ids = [corpus_ids[idx] for idx in candidate_indices]
        candidate_texts = [corpus_texts[idx] for idx in candidate_indices]

        # --- STAGE 2: Reranking (BGE-Reranker-V2-M3) ---
        # Format inputs as pairs: [[query, doc1], [query, doc2], ...]
        rerank_pairs = [[query_text, doc_text] for doc_text in candidate_texts]

        # Get cross-encoder scores
        rerank_scores = reranker.predict(rerank_pairs)

        # Sort documents by reranker score in descending order
        reranked_indices = np.argsort(rerank_scores)[::-1]
        final_ranked_doc_ids = [candidate_doc_ids[idx] for idx in reranked_indices]


        # --- Metric Calculation ---
        ndcg_val = compute_ndcg_at_k(final_ranked_doc_ids, qrels[qid], k=top_k_ndcg)
        ndcg_scores.append(ndcg_val)

        results.append({
            "qid": qid,
            "query_text": query_text,
            "candidate_doc_ids": candidate_doc_ids,
            "final_ranked_doc_ids": final_ranked_doc_ids,
            "gold_docs": qrels[qid],
            "ndcg_val": ndcg_val
        })

        # print(f"Query: '{query_text}' | NDCG@{top_k_ndcg}: {ndcg_val:.4f}")

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"\nOverall Mean NDCG@{top_k_ndcg}: {mean_ndcg:.4f}")
    print(f"Saving into {results_path}")
    pd.DataFrame(results).to_csv(results_path, index=False)
    return mean_ndcg


def load_ds(name):
    dataset_path = "../data/dataset"
    questions = pd.read_json(f"{dataset_path}/{name}/ifeval-{name}.json")
    passages = pd.read_json(f"{dataset_path}/{name}/passages-{name}.jsonl", lines=True)
    return questions, passages


def prepare_evaluation_data(df_questions, df_documents):
    queries = {}
    corpus = {}
    qrels = {}

    print("Building corpus...")
    for _, row in df_documents.iterrows():
        doc_id = str(row['id']).strip()

        # Combine title and contents for richer document representation
        contents = str(row['contents']) if pd.notna(row['contents']) else ""

        # Store in corpus
        if 'title' in row:
            title = str(row['title']) if pd.notna(row['title']) else ""
            corpus[doc_id] = f"{title}\n{contents}".strip()
        else:
            corpus[doc_id] = contents

    print("Building queries and qrels...")
    for _, row in df_questions.iterrows():
        q_id = str(row['id']).strip()
        query_text = str(row['question']).strip()

        # Add to queries dict
        queries[q_id] = query_text

        # Initialize the qrels dict for this query
        qrels[q_id] = {}

        # Parse the 'context' column to get relevant document IDs
        context_data = row['context']
        relevant_docs = []

        # Handle the case where the context might be loaded as a string representation of a list
        if isinstance(context_data, str):
            context_data = context_data.strip()
            if context_data.startswith('[') and context_data.endswith(']'):
                # Remove brackets and split by comma
                inner_str = context_data[1:-1]
                if inner_str.strip():
                    # Split, clean up spaces and quotes
                    relevant_docs = [doc.strip().strip("'\"") for doc in inner_str.split(',')]
        elif isinstance(context_data, list):
            relevant_docs = context_data

        # Assign relevance score of 1 to all documents found in the context list
        for doc_id in relevant_docs:
            if doc_id in corpus: # Ensure the document actually exists in your corpus
                qrels[q_id][doc_id] = 1

    # Remove queries from qrels that have no relevant documents (e.g., the "refuse: True" ones)
    # NDCG calculation requires at least one relevant document to be meaningful.
    qrels = {qid: docs for qid, docs in qrels.items() if len(docs) > 0}

    print(f"Generated {len(corpus)} documents.")
    print(f"Generated {len(queries)} queries.")
    print(f"Generated {len(qrels)} queries with valid relevance judgments (qrels).")

    return queries, corpus, qrels

def main():
    # load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models on: {device.upper()}")

    # Stage 1: Dense Retriever (BGE-M3)
    print("Loading BGE-M3 Retriever...")
    retriever = SentenceTransformer('BAAI/bge-m3', device=device)

    # Stage 2: Cross-Encoder Reranker (BGE-Reranker-v2)
    print("Loading BGE-Reranker-v2-M3...")
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)


    # load data
    dataset_path = "../data/dataset"
    res_dir = "..data/rag"
    files = list(os.listdir(dataset_path))
    for name in tqdm(files):
        if os.path.exists(f"{res_dir}/{name}.csv"):
            print(f"Skipping: {name}, already processed")
        else:
            print(f"Processing: {name}")
            q, p = load_ds(name)
            queries, corpus, qrels = prepare_evaluation_data(q, p)
            print(f"Evaluating...")
            evaluate_pipeline(retriever, reranker, queries, corpus, qrels, top_k_retrieve=100, top_k_ndcg=10, results_path=f"{res_dir}/{name}.csv")

if __name__ == "__main__":
    main()
