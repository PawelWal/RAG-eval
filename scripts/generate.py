import os
import ast
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import os
import ast
import click
import operator
import pandas as pd
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. State Definition
# ==========================================
class RAGState(TypedDict):
    # --- Input parameters ---
    retrieval_path: str
    queries_path: str
    documents_path: str
    llm_params: Dict[str, Any]
    prompt_path: str

    # --- Populated by Init Node ---
    qids: Optional[List[str]]                # List of queries to process
    retrieval_df: Optional[pd.DataFrame]
    queries_df: Optional[pd.DataFrame]
    documents_df: Optional[pd.DataFrame]
    llm: Optional[Any]              # Storing the instantiated LLM
    prompt: Optional[Any]
    embeddings_model: Optional[Any] # Stores the embedding model

    # --- Loop State ---
    current_idx: int

    # Annotated with operator.add means that when a node returns {"results": [new_item]},
    # LangGraph will APPEND it to the existing list rather than overwriting it.
    results: Annotated[List[Dict[str, Any]], operator.add]
    current_row: Optional[List[Any]]

# ==========================================
# 2. Initialization Node
# ==========================================
def init_system(state: RAGState) -> Dict[str, Any]:
    print("[Node: init_system] Loading files and initializing LLM...")

    # Load dataframes once
    retrieval_df = pd.read_csv(state["retrieval_path"])
    queries_df = pd.read_json(state["queries_path"])
    documents_df = pd.read_json(state["documents_path"], lines=True)

    # Initialize the LLM once
    llm = ChatOpenAI(**state["llm_params"])
    print("[Node: init_system] Loading Sentence Transformer model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs = {'device': 'cuda:0'}
    )

    with open(state.get("prompt_path", "../prompts/base-prompt.jinja"), "r", encoding="utf-8") as f:
        template_str = f.read()

    prompt = PromptTemplate(
        template=template_str,
        template_format="jinja2",
        input_variables=["docs", "question"]
    )


    # Pass loaded objects to the workflow state
    return {
        "retrieval_df": retrieval_df,
        "queries_df": queries_df,
        "documents_df": documents_df,
        "qids": queries_df['qid'].to_list() if 'qid' in queries_df.columns else queries_df['id'].to_list(),
        "llm": llm,
        "prompt": prompt,
        "embeddings_model": embeddings_model,
        "current_idx": 0 # Start loop at index 0
    }

# ==========================================
# 3. Processing & Generation Node
# ==========================================
def process_and_generate(state: RAGState) -> Dict[str, Any]:
    # Identify which query we are processing in this loop iteration
    idx = state["current_idx"]
    qid = state["qids"][idx]
    print(f"[Node: process_and_generate] Processing {qid} ({idx + 1}/{len(state['qids'])})...")

    # Extract resources from state
    retrieval_df = state["retrieval_df"]
    queries_df = state["queries_df"]
    documents_df = state["documents_df"]
    llm = state["llm"]
    prompt = state["prompt"]

    # --- A. Data Preparation ---
    retrieval_row = retrieval_df[retrieval_df['qid'] == qid].iloc[0]

    doc_ids_raw = retrieval_row['final_ranked_doc_ids']
    ranked_doc_ids = ast.literal_eval(doc_ids_raw) if isinstance(doc_ids_raw, str) else doc_ids_raw
    top_5_ids = ranked_doc_ids[:5]

    query_row = queries_df[queries_df['id'] == qid].iloc[0]
    question = query_row['question']
    gold_answer = query_row['gold_answer']
    expect = query_row['expect']

    docs_contents = []
    for doc_id in top_5_ids:
        doc_row = documents_df[documents_df['id'] == doc_id].iloc[0]
        if not doc_row.empty:
            # Combine title and contents for richer document representation
            contents = str(doc_row['contents']) if pd.notna(doc_row['contents']) else ""

            # Store in corpus
            if 'title' in doc_row:
                title = str(doc_row['title']) if pd.notna(doc_row['title']) else ""
                contents = f"{title}\n{contents}".strip()

            docs_contents.append(contents)

    chain = prompt | llm
    response = chain.invoke({
        "docs": docs_contents,
        "question": question
    })

    # Pack the result
    new_result = {
        "qid": qid,
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": response.content,
        "docs": top_5_ids,
        "expect": expect,
        "gold_answer": gold_answer,
        "current_docs": docs_contents
    }

    # Return the new result (LangGraph will append it due to operator.add)
    # and increment the loop counter
    return {
        "current_row": new_result,
    }

def evaluate_similarity(state: RAGState) -> Dict[str, Any]:
    gold = state["current_row"]["gold_answer"]
    generated = state["current_row"]["generated_answer"]
    embeddings_model = state["embeddings_model"]
    llm = state["llm"]

    # Create vector embeddings for both strings
    vec_gold = embeddings_model.embed_query(gold)
    vec_gen = embeddings_model.embed_query(generated)

    # Calculate Cosine Similarity using NumPy
    cos_sim = np.dot(vec_gold, vec_gen) / (np.linalg.norm(vec_gold) * np.linalg.norm(vec_gen))

    # ---------------------------------------------------------
    # METRIC 2: AutoAIS (LLM-as-a-judge)
    # ---------------------------------------------------------
    # Convert the list of docs into a single numbered string for the prompt
    docs = state["current_row"]["current_docs"]
    docs_text = "\n".join([f"Doc {i}: {doc}" for i, doc in enumerate(docs)])

    # AutoAIS Prompt: We instruct the LLM to act as a strict evaluator
    with open("../prompts/ais.txt", "r", encoding="utf-8") as f:
        ais_template = f.read()

    ais_prompt = PromptTemplate(
        template=ais_template,
        input_variables=["docs_text", "generated"]
    )

    # Run the judge
    ais_chain = ais_prompt | llm
    ais_response = ais_chain.invoke({"docs_text": docs_text, "generated": generated})

    # Parse the output safely
    try:
        ais_score = int(ais_response.content.strip())
        if ais_score not in [0, 1]:
            ais_score = 0
    except ValueError:
        ais_score = 0 # Default to 0 if the model failed to format the output

    # Pack the result
    new_result = {
        **state["current_row"],
        "ais_score": ais_score,
        "cosine_similarity": float(cos_sim) # Convert np.float to native Python float
    }

    # Append to results list and increment the loop index
    return {
        "results": [new_result],
        "current_idx": state["current_idx"] + 1
    }

# ==========================================
# 4. Router (Conditional Edge)
# ==========================================
def should_continue(state: RAGState) -> str:
    # If we haven't processed all queries, loop back. Otherwise, finish.
    if state["current_idx"] < len(state["qids"]):
        return "process_and_generate"
    return END

# ==========================================
# 5. Build Workflow
# ==========================================
workflow = StateGraph(RAGState)

workflow.add_node("init", init_system)
workflow.add_node("process_and_generate", process_and_generate)
workflow.add_node("evaluate_similarity", evaluate_similarity) # Add new node

# Define the flow
workflow.add_edge(START, "init")
workflow.add_edge("init", "process_and_generate")
workflow.add_edge("process_and_generate", "evaluate_similarity") # Generation passes to Evaluation

# Router now lives after Evaluation
workflow.add_conditional_edges("evaluate_similarity", should_continue)

app = workflow.compile()

# ==========================================
# 6. Usage Example
# ==========================================
def test():
    initial_state = {
        "retrieval_path": "retrieval.csv",
        "queries_path": "queries.csv",
        "documents_path": "documents.csv",
         "llm_params": {
            "model_name": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
            "temperature": 0.1,
            "max_tokens": 500,
        },
    }

    print("Starting LangGraph Workflow...")
    final_state = app.invoke(initial_state)

    print("\n--- Final Results ---")
    for res in final_state["results"]:
        print(f"\nQID: {res['qid']}")
        print(f"Question: {res['question']}")
        print(f"Generated: {res['generated_answer']}")

@click.command()
@click.option('--ds-name')
def main(ds_name):
    initial_state = {
        "retrieval_path": f"/mnt/data/pwalkow/rag/{ds_name}.csv",
        "queries_path": f"../data/dataset/{ds_name}/ifeval-{ds_name}.json",
        "documents_path": f"../data/dataset/{ds_name}/passages-{ds_name}.jsonl",
         "llm_params": {
             "model_name": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
            "temperature": 0.3,
            "max_tokens": 500,

        },
    }

    # print("Starting LangGraph Workflow...")
    # final_state = app.invoke(initial_state)

    # 1. Get the total number of queries for the progress bar
    df = pd.read_csv(initial_state['retrieval_path'])
    total_queries = len(df)

    # We will collect the final results here as they stream in
    final_results = []

    # 2. Initialize tqdm context manager
    with tqdm(total=total_queries, desc="Generating Responses") as pbar:

        # 3. Use app.stream instead of app.invoke
        # stream_mode="updates" yields exactly what each node returns as it finishes
        for step in app.stream(initial_state, stream_mode="updates"):

            # 4. Check if the step that just finished is our processing node
            if "evaluate_similarity" in step:
                # Update the progress bar by 1
                pbar.update(1)

                # Extract the newly generated result from this step
                node_output = step["evaluate_similarity"]
                if "results" in node_output:
                    # Extend our final list with the new result
                    final_results.extend(node_output["results"])

    print("\n--- Final Results ---")
    res_name = f"../data/rag_generation/{ds_name}.csv"
    pd.DataFrame(final_results).to_csv(res_name, index=False)
    print(f"\nOK saved {res_name}")

if __name__ == "__main__":
    main()
