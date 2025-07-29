import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import openai
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests
import zlib
from tqdm import tqdm
import re
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
from IPython.display import clear_output
import pickle
import random
import pprint
pp = pprint.PrettyPrinter(depth=2)
import urllib.parse
from sklearn.metrics import confusion_matrix
import inspect
from pathlib import Path

memory = Memory("cachegpt", verbose=0)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version




client = openai.OpenAI(api_key=openai.api_key)

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    # print('Calling OpenAI...')
    response = client.chat.completions.create(
        temperature=temperature,
        # deployment_id=chat_deployment_id,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    return response.choices[0].message.content.strip()


@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)


def get_gpt35():
    chat_model_id = 'gpt-35-turbo' if openai.api_type == 'azure' else 'gpt-3.5-turbo'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    resp = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    

def drug_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DRUG_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DRUG_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Drugs"]
    except:
        return None
    

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

######################################################################################

def decode_and_decompress(data_str):
    try:
        # Remove the initial b' and final '
        data_str = data_str[2:-1]
        # Replace escaped sequences
        data_bytes = bytes(data_str, "utf-8").decode("unicode_escape").encode("ISO-8859-1")
        # Decompress the data
        decompressed_data = zlib.decompress(data_bytes)
        # Convert to JSON
        json_data = json.loads(decompressed_data)
        return json_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def serialize_result(result):
    """Compress and serialize a JSON-compatible object."""
    return zlib.compress(json.dumps(result).encode())

def deserialize_result(serialized_result):
    """Decompress and deserialize the stored context."""
    return json.loads(zlib.decompress(serialized_result).decode())

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# def extract_answer(llm_answer_prompt_test):
#     llm_answer_prompt_test = llm_answer_prompt_test.replace('```json\n', '').replace('```', '')
#     llm_answer = json.loads(llm_answer_prompt_test)
#     return llm_answer['answer']

def make_readable(text):
    # Use regular expressions for case-insensitive replacements
    text = re.sub(r"biolink:disease", "", text, flags=re.IGNORECASE)
    text = re.sub(r"biolink:gene", "", text, flags=re.IGNORECASE)
    text = re.sub(r"biolink:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_", " ", text)
    text = re.sub(r":", "", text)
    text = re.sub(r"INFORES", "", text, flags=re.IGNORECASE)
    text = ' '.join(text.split())  # Remove extra whitespaces
    return text


def get_bte_api_resp(base_uri, query):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(base_uri, json=query, headers=headers)
    return response.json()

def create_sentence(row):
    return f"{row['n0_description']} is associated with gene {row['n1_description']}, and the source for this information is {row['resource_id']}."

def clean_description(desc):
    """
    Remove any leading entity-type words like 'SmallMolecule', 'ChemicalEntity',
    'Drug', etc. Optionally you could replace them with more descriptive terms
    if you want.
    """

    known_prefixes = ["SmallMolecule ", "ChemicalEntity ", "Drug ", "MolecularMixture ", "ChemicalMicture ","ComplexMolecularMixture ", "MolecularEntity ", "NucleicAcidEntity "]
    
    for prefix in known_prefixes:
        if desc.startswith(prefix):
            return desc[len(prefix):]  # e.g. remove "SmallMolecule "
    
    return desc  # return unchanged if no prefix matches

def objects_to_text(objs):
    return ", ".join(objs)

####################################################################################################

def get_context_for_id(doid, doid_type):

    # Parquet file which stores BTE results for the IDs
    bte_parquet = pd.read_parquet("data/analysis_results/bte_results.parquet")
    
    # Check if the DOID exists; if not, return empty lists.
    if doid not in bte_parquet["entity_id"].values:
        print(f"ID {doid} not found in the DataFrame.")
        return [], []
    
    try:
        context_serialized = bte_parquet.loc[bte_parquet["entity_id"] == doid, "context"].iloc[0]
        node_context = deserialize_result(context_serialized)
    except Exception as e:
        print(f"Exception for DOID {doid}: {str(e)}")
        return None

    # -------------------------------
    # Process the knowledge graph data
    # -------------------------------
    nbr_nodes = []
    nbr_edges = []
    nodes = node_context.get("message", {}).get("knowledge_graph", {}).get("nodes", {})
    edges = node_context.get("message", {}).get("knowledge_graph", {}).get("edges", {})

    # Processing node information
    for node_id, node_info in nodes.items():
        categories = ", ".join(node_info.get("categories", []))
        node_name = node_info.get("name", "")
        nbr_nodes.append((node_id, categories, node_name))

    # Processing edge information
    for edge_id, edge_info in edges.items():
        predicate = edge_info.get("predicate", "")
        subject = edge_info.get("subject", "")
        object_ = edge_info.get("object", "")
        # Get the primary source if available; otherwise, join all resource_ids.
        primary_source = next(
            (source.get("resource_id") for source in edge_info.get("sources", [])
             if source.get("resource_role") == "primary_knowledge_source"),
            None
        )
        if primary_source:
            sources = primary_source
        else:
            sources = ", ".join(source.get("resource_id", "") for source in edge_info.get("sources", []))
        nbr_edges.append((subject, predicate, object_, sources))    

    # Create DataFrames for nodes and edges
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_id", "categories", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["subject", "predicate", "object", "sources"])
    if not nbr_edges_df.empty:
        nbr_edges_df['sources'] = nbr_edges_df['sources'].str.upper()

    # Merge nodes and edges 
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="subject", right_on="node_id",
                       suffixes=('', '_node')).drop("node_id", axis=1)
    merge_1["subject_description"] = merge_1["categories"] + " " + merge_1["node_name"]
    merge_1.drop(["categories", "node_name"], axis=1, inplace=True)

    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="object", right_on="node_id", how="left",
                       suffixes=('', '_node')).drop("node_id", axis=1)
    merge_2["object_description"] = merge_2["categories"] + " " + merge_2["node_name"]
    merge_2.drop(["categories", "node_name"], axis=1, inplace=True)
    merge_2 = merge_2[merge_2["object_description"] != ""]
    merge_2 = merge_2.map(make_readable) 
    merge_2_filtered = merge_2

    # Create a context strings
    merge_2_filtered['context'] = merge_2_filtered.apply(
        lambda row: (
            f"{row['subject_description']} is {row['predicate']} "
            f"{row['object_description']}."
        ),
        axis=1
    )
    merge_2_filtered["subject_clean"] = merge_2_filtered["subject_description"].apply(clean_description)
    merge_2_filtered["object_clean"] = merge_2_filtered["object_description"].apply(clean_description)

    merge_2_filtered["context"] = merge_2_filtered.apply(
        lambda row: f"{row['subject_clean']} is {row['predicate']} {row['object_clean']}.",
        axis=1
    )
    # Group by subject and predicate
    grouped = (
    merge_2_filtered
    .groupby(["subject_clean", "predicate"])["object_clean"]
    .apply(list)
    .reset_index()
    )

    grouped["object_list_str"] = grouped["object_clean"].apply(objects_to_text)

    grouped["context_grouped"] = grouped.apply(
        lambda row: (
            f"{row['subject_clean']} is {row['predicate']} {row['object_list_str']}."
        ),
        axis=1
    )
    print("doid type:", doid_type)
    # Filter based on the doid_type
    if doid_type=="drug":
        print("drug")
        final_df = grouped[["context_grouped"]].drop_duplicates()
        node_context_final = final_df['context_grouped'].tolist()
    elif doid_type=="disease":
        print("disease")
        final_df = grouped[grouped["predicate"]=="subject of treatment application or study for treatment by"][["context_grouped"]].drop_duplicates()
        node_context_final = final_df['context_grouped'].tolist()

    return node_context_final

def retrieve_context_v17(question, drug_id, drug_name, disease_id, disease_name, embedding_function, context_sim_threshold, context_sim_min_threshold, context_volume):
    
    # question
    print(question)
    question_embedding = embedding_function.embed_query(question)

    # Retrieve and process context for drug_id
    print("Processing Drug ID:", drug_id)
    drug_context = get_context_for_id(drug_id, doid_type ="drug")

    if drug_context:
        # Calculate similarity for the drug branch.
        drug_embeddings = embedding_function.embed_documents(drug_context)
        drug_similarities = [
            cosine_similarity(
                np.array(question_embedding).reshape(1, -1), 
                np.array(context_embedding).reshape(1, -1)
            )[0][0] for context_embedding in drug_embeddings
        ]
        drug_similarity_indices = sorted(
            [(sim, idx) for idx, sim in enumerate(drug_similarities)], 
            key=lambda x: x[0], 
            reverse=True
        )
        drug_percentile_threshold = np.percentile([s[0] for s in drug_similarity_indices], context_sim_threshold)
        high_similarity_drug_indices = [
            idx for sim, idx in drug_similarity_indices 
            if sim > drug_percentile_threshold and sim > context_sim_min_threshold
        ]
        high_similarity_drug_context = [drug_context[idx] for idx in high_similarity_drug_indices]
    else:
        print("Drug context or data frame is empty; skipping similarity computations for drug.")
        high_similarity_drug_context = []

    # Retrieve and process context for disease_id
    print("Processing Disease ID:", disease_id)
    disease_context = get_context_for_id(disease_id, doid_type = "disease")
    if disease_context:
        # Calculate similarity for the disease branch.
        disease_embeddings = embedding_function.embed_documents(disease_context)
        disease_similarities = [
            cosine_similarity(
                np.array(question_embedding).reshape(1, -1), 
                np.array(context_embedding).reshape(1, -1)
            )[0][0] for context_embedding in disease_embeddings
        ]
        disease_similarity_indices = sorted(
            [(sim, idx) for idx, sim in enumerate(disease_similarities)], 
            key=lambda x: x[0], 
            reverse=True
        )
        disease_percentile_threshold = np.percentile([s[0] for s in disease_similarity_indices], context_sim_threshold)
        high_similarity_disease_indices = [
            idx for sim, idx in disease_similarity_indices 
            if sim > disease_percentile_threshold and sim > context_sim_min_threshold
        ]
        high_similarity_disease_context = [disease_context[idx] for idx in high_similarity_disease_indices]
    else:
        print("Disease context or data frame is empty; skipping similarity computations for disease.")
        high_similarity_disease_context = []

    # Combine high-similarity contexts into output strings.
    combined_context = (
        f"Context for Drug {drug_name}:\n\n" +
        "\n".join(high_similarity_drug_context) +
        f"\n\nContext for Disease {disease_name}:\n\n" +
        "\n".join(high_similarity_disease_context)
    )
    node_context_extracted = combined_context + "\n"
    
    # Build combined context strings using the full lists returned from get_context_for_id.
    combined_context_str = (
    f"Context for Drug {drug_name}:\n\n" +
    "\n".join(drug_context if drug_context else []) +  
    f"\n\nContext for Disease {disease_name}:\n\n" +
    "\n".join(disease_context if disease_context else [])  
)
    print("few lines of combined_context_str", combined_context_str[:10])
    print("NEW 9")

    return node_context_extracted, combined_context_str


def truncate_context(text, max_tokens, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the string
    tokens = encoding.encode(text)
    token_count = len(tokens)
    # print("type of token_count:", type(token_count))
    print(f"Original token count: {token_count}")

    if token_count > max_tokens:
        print(f"Truncating to {max_tokens} tokens.")
        tokens = tokens[:max_tokens]  # Truncate to the allowed token count
        print(f"Truncated token count: {len(tokens)}")
        truncated_text = encoding.decode(tokens)  # Detokenize back to text
        return truncated_text, len(tokens)

    return text, token_count  # If within limits, return the original text


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def retrieve_output_from_llm_v17(
    question, node_context_extracted, combined_context_str, llm_type, system_prompt
):
    # Token limit
    TOKEN_LIMIT = 128000
    TRUNCATED_TOKEN_COUNT = 127000

    

    # Truncate contexts if needed
    node_context_extracted_trunc, node_context_tokens_trunc = truncate_context(node_context_extracted, TRUNCATED_TOKEN_COUNT, "o200k_base")
    
    
    combined_context_str_trunc, combined_context_tokens_trunc = truncate_context(combined_context_str, TRUNCATED_TOKEN_COUNT, "o200k_base")
    

    # Generate outputs based on LLM type
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(
            config_data["LLAMA_MODEL_NAME"],
            config_data["LLAMA_MODEL_BRANCH"],
            config_data["LLM_CACHE_DIR"],
            stream=True,
            method=llama_method,
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt_1 = "Context: " + node_context_extracted_trunc + "\n" + "Question: " + question
        output_1 = get_GPT_response(
            enriched_prompt_1, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"]
        )
        print("done 1",output_1)

        enriched_prompt_NC_final = "Context: " + combined_context_str_trunc + "\n" + "Question: " + question
        output_NC_final = get_GPT_response(
            enriched_prompt_NC_final, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"]
        )
        print("done 2", output_NC_final)


    return (
        output_1,
        output_NC_final,
        node_context_extracted_trunc,
        combined_context_str_trunc,
        node_context_extracted,combined_context_str
    )


def retrieve_combined_from_llm(
    question,
    combined_context_str,
    llm_type,
    system_prompt
):
    # truncate the combined context
    TRUNCATED_TOKEN_COUNT = 127000
    combined_trunc, _ = truncate_context(
        combined_context_str,
        TRUNCATED_TOKEN_COUNT,
        "o200k_base"
    )

    # single LLM call on combined
    enriched_prompt = (
        "Context: " + combined_trunc + "\n"
        "Question: " + question
    )
    output_NC_final = get_GPT_response(
        enriched_prompt,
        system_prompt,
        llm_type,
        llm_type,
        temperature=config_data["LLM_TEMPERATURE"]
    )
    return output_NC_final, combined_trunc


def retrieve_node_from_llm(
    question,
    node_context_extracted,
    llm_type,
    system_prompt
):
    # truncate the node context
    TRUNCATED_TOKEN_COUNT = 127000
    node_trunc, _ = truncate_context(
        node_context_extracted,
        TRUNCATED_TOKEN_COUNT,
        "o200k_base"
    )

    # single LLM call on node context
    enriched_prompt = (
        "Context: " + node_trunc + "\n"
        "Question: " + question
    )
    output_sim = get_GPT_response(
        enriched_prompt,
        system_prompt,
        llm_type,
        llm_type,
        temperature=config_data["LLM_TEMPERATURE"]
    )
    return output_sim, node_trunc



###################################################################################################


def extract_answer(text):
    """
    Robustly extract the value associated with key 'answer'.
    Handles raw JSON, JSON wrapped in markdown ``` blocks,
    or loose 'answer: VALUE' lines. Returns None on failure.
    """
    if not isinstance(text, str):
        return None

    txt = text.strip()

    # 1) Remove markdown code‑block fences (``` or ```json)
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)

    # 2) Quick route: is it valid JSON now?
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    except Exception:
        pass

    # 3) Fallback: look for 'answer: <token>'
    m = re.search(r"answer\s*[:=]\s*['\"]?([\w\-]+)['\"]?", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def process_llm_mechanistic_answers(
        df, 
        col_answer_test="llm_answer_prompt_test", 
        col_answer_out="llm_answer_prompt", 
        col_protein_symbol="protein_gene_symbol", 
        col_is_correct="is_correct_in_llm", 
        col_protein_str="protein_gene_symbol_str", 
        extract_answer_function=None
    ):
    
    if extract_answer_function is None:
        raise ValueError("Please provide an extract_answer_function.")

    # 1) Extract the bare answer string
    df[col_answer_out] = df[col_answer_test].apply(extract_answer_function)

    # 2) Turn the stored protein symbols into a Python list
    df[col_protein_symbol] = df[col_protein_symbol].apply(ast.literal_eval)

    # 3) Check correctness by seeing if the predicted answer is
    #    a substring of any of the true symbols (case-insensitive)
    df[col_is_correct] = df.apply(
        lambda row: any(
            str(row[col_answer_out]).lower() in str(sym).lower()
            for sym in row[col_protein_symbol]
        ),
        axis=1
    )

    # 4) Build a human-readable string of the symbol list
    df[col_protein_str] = df[col_protein_symbol].apply(
        lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
    )

    return df

def build_thresholded_context_df(
    questions_df,
    embedding_function,
    thresholds,
    min_threshold,
    context_volume,
    save_path
):
    # 1) load or initialize
    if os.path.exists(save_path):
        out_df = pd.read_csv(save_path)
    else:
        base_cols = [
            "id", "count", "question", "drug_name", "disease_name",
            "Drug_MeshID", "disease", "protein", "protein_name"
        ]
        thr_cols = []
        for thr in thresholds:
            thr_cols += [
                f"node_context_extracted_{thr}_compressed",
                f"combined_context_str_{thr}_compressed"
            ]
        out_df = pd.DataFrame(columns=base_cols + thr_cols)

    processed = set(out_df["question"])

    # 2) iterate questions
    for _, row in questions_df.iterrows():
        qid   = row["id"]
        question      = row["question"]
        drug_id       = row["Drug_MeshID"]
        drug_name     = row["drug_name"]
        disease_id    = row["disease"]
        disease_name  = row["disease_name"]
        count         = row["count"]
        protein  = row["protein_gene_symbol"]
        protein_name = row["protein_name"]

        if question in processed:
            continue

        # prepare row
        row_dict = {
            "id": qid,
            "count": count,
            "question": question,
            "drug_name": drug_name,
            "disease_name": disease_name,
            "Drug_MeshID": drug_id,
            "disease": disease_id,
            "protein": protein,
            "protein_name": protein_name,
        }

        # 3) loop thresholds
        for thr in thresholds:
            try:
                node_ctx, combined_ctx = retrieve_context_v17(
                    question, drug_id, drug_name,
                    disease_id, disease_name,
                    embedding_function,
                    thr,
                    min_threshold,
                    context_volume
                )
                # compress JSON strings
                nc_json = json.dumps(node_ctx)
                cc_json = json.dumps(combined_ctx)
                row_dict[f"node_context_extracted_{thr}_compressed"]   = zlib.compress(nc_json.encode("utf-8"))
                row_dict[f"combined_context_str_{thr}_compressed"]    = zlib.compress(cc_json.encode("utf-8"))

            except Exception as e:
                # on error, store None
                row_dict[f"node_context_extracted_{thr}_compressed"]   = None
                row_dict[f"combined_context_str_{thr}_compressed"]    = None
                print(f"retrieve_context_v17 error at threshold {thr}: {e}")

        # 4) append & save
        out_df = pd.concat([out_df, pd.DataFrame([row_dict])], ignore_index=True)
        out_df.to_csv(save_path, index=False)
        print(f"Built contexts for question {qid}")

    print("All done — contexts saved to", save_path)
    return out_df

import ast

def extract_first_from_list_column(df, 
                                    source_col="protein", 
                                    target_col="protein_gene_symbol_str"):
    
    df[target_col] = df[source_col].apply(ast.literal_eval)
    df[target_col] = df[target_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

    return df


def process_drugmechDB_v2(df, extract_answer_function):
    df = df.copy()                         # avoid mutating caller’s frame
    output_cols = [c for c in df.columns if c.startswith("output_")]

    # 1. Remove rows where any output_ is missing entirely
    df = df.dropna(subset=output_cols, how="any")

    # 2. Parse every output_ column
    for col in output_cols:
        df[col] = df[col].apply(extract_answer_function)

    # 3. Remove rows where parsing failed in *all* output_ columns
    #    (keep row if at least one model produced an answer)
    df = df.dropna(subset=output_cols, how="all")

    # 4. Normalise 'protein_gene_symbol_str'
    if df["protein_gene_symbol_str"].apply(lambda x: isinstance(x, list)).any():
        df["protein_gene_symbol_str"] = df["protein_gene_symbol_str"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
        )

    # 5. Case‑insensitive membership tests
    pg_col_lower = df["protein_gene_symbol_str"].str.lower()

    for col in output_cols:
        result_col = f"result_in_{col}"
        df[result_col] = df.apply(
            lambda row: str(row[col]).lower() in pg_col_lower.loc[row.name],
            axis=1
        )

    return df


###################PLOTS#########################################

import os, re, pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_context_performance(
    df               : pd.DataFrame,
    nc_col           : str = "result_in_output_NC",
    sim_prefix       : str = "result_in_output_sim_",
    save_dir         : str | os.PathLike = "figures/mechanistic_genes",
    model_label      : str = "gpt‑4o‑mini",
    palette_name     : str = "colorblind",
    dpi              : int = 300,
):
   
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    
    pat = re.compile(rf"^{re.escape(sim_prefix)}(\d+)$")
    sims = {int(m.group(1)): c for c in df.columns
            for m in [pat.match(c)] if m}
    if not sims:
        raise ValueError(f"No cols with prefix '{sim_prefix}'")
    thrs = sorted(sims)

    # -------- pick base colour --------
    cb = sns.color_palette(palette_name, 10)
    base = cb[1] if "mini" in model_label.lower() else cb[0]   # orange / blue

   
    rows = []
    for t, col in sims.items():
        rows.append({"ctx": f"> {t}th", "thr": t,
                     "acc": df[col].mean()*100,
                     "true": df[col].sum(),
                     "false": (~df[col]).sum()})
    rows.append({"ctx": "Full Context", "thr": None,
                 "acc": df[nc_col].mean()*100,
                 "true": df[nc_col].sum(),
                 "false": (~df[nc_col]).sum()})
    met = pd.DataFrame(rows)

    def save(fig, name):
        fig.savefig(pathlib.Path(save_dir)/f"{model_label}_{name}.svg",
                    format="svg", dpi=dpi, bbox_inches="tight")

    # ==============================================================
    #  FIGURE 1 • Accuracy bar with lighter shades + rotated labels
    # ==============================================================
    
    order = met.sort_values("thr", na_position="first")
    shades  = sns.light_palette(base, n_colors=len(order), reverse=True)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    # ax1.bar(order["ctx"], order["acc"], color=shades)
    ax1.bar(order["ctx"], order["acc"], color=base)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Context lines above a certain percentile")
    ax1.set_ylim(0, 100)
    ax1.set_title(f"{model_label}: Accuracy across contexts")
    ax1.set_xticklabels(order["ctx"], rotation=45, ha="right")
    for x, v in enumerate(order["acc"]):
        ax1.text(x, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    sns.despine()
    save(fig1, "accuracy_bar")
    plt.show()

    # ==============================================================
    #  FIGURE 2 • Accuracy‑vs‑threshold curve (single colour)
    # ==============================================================
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    thr_df = met.dropna(subset=["thr"]).sort_values("thr")
    ax2.plot(thr_df["thr"], thr_df["acc"], marker="o", linewidth=2, color=base)
    full_acc = met.loc[met["ctx"] == "Full Context", "acc"].values[0]
    ax2.axhline(full_acc, color=base, linestyle="--", linewidth=2,
                label="Full Context")
    ax2.set_xlabel("Percentile cut‑off")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(thrs)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"{model_label}: Accuracy vs percentile")
    ax2.legend()
    sns.despine()
    save(fig2, "accuracy_curve")
    plt.show()

    # ==============================================================
    #  FIGURE 3 • Stacked counts (single colour front bar)
    # ==============================================================
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.bar(order["ctx"], order["true"], color=base, label="Correct")
    ax3.bar(order["ctx"], order["false"], bottom=order["true"],
            color="lightgrey", label="Incorrect")
    ax3.set_ylabel("Number of answers")
    ax3.set_xlabel("Context lines above a certain percentile")
    ax3.set_title(f"{model_label}: Prediction counts")
    ax3.set_xticklabels(order["ctx"], rotation=45, ha="right")
    ax3.legend()
    sns.despine()
    save(fig3, "stacked_counts")
    plt.show()

    print(f"Figures saved to → {save_dir.resolve()}")


# ---------------------------------------------------------------
#  Publication‑quality LLM‑vs‑RAG comparison with model‑colours
# ---------------------------------------------------------------
import pathlib, os, colorsys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import to_rgb

def evaluate_llm_vs_rag_multi(
    df_llm      : pd.DataFrame,
    df_rag      : pd.DataFrame,
    id_cols     : list[str],
    llm_col     : str   = "is_correct_in_llm",
    rag_cols    : list[str] | None = None,   
    model_name  : str   = "gpt‑4o",          # selects orange / blue theme
    save_dir    : str   | os.PathLike = "figures_llm_vs_rag",
    dpi         : int   = 300
) -> pd.DataFrame:

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------- 
    cb                = sns.color_palette("colorblind", 10)
    base_colour       = cb[1] if "mini" in model_name.lower() else cb[0]  # orange / blue
    lighter_colour    = sns.light_palette(base_colour, n_colors=4)[0]     # same hue, lighter
    hatch_pattern     = "///"

    def _save(fig, tag):                       # helper for SVG output
        fig.savefig(save_dir / f"{model_name}_{tag}.svg",
                    format="svg", dpi=dpi, bbox_inches="tight")

    def _seq_cmap():                           # sequential cmap for heat‑map
        return sns.light_palette(base_colour, as_cmap=True, reverse=True)

    # -----------------------------------  
    df = pd.merge(df_llm, df_rag, how="inner", on=id_cols,
                  suffixes=("_llm", "_rag"))

    if rag_cols is None:                       # auto‑detect if not supplied
        rag_cols = [c for c in df.columns if c.startswith("result_in_output_")]
        if not rag_cols:
            raise ValueError("No RAG result columns were found!")

    # -----------------------------------  
    n_total = len(df)
    llm_acc = df[llm_col].mean() * 100
    llm_cnt = df[llm_col].value_counts().reindex([False, True], fill_value=0)

    summary_rows = []

    # -----------------------------------  iterate chosen RAG columns
    for col in rag_cols:
        rag_acc = df[col].mean() * 100
        rag_cnt = df[col].value_counts().reindex([False, True], fill_value=0)
        lift    = rag_acc - llm_acc

        # ================= FIG 1 • Accuracy ==========================
        # portrait orientation
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        # clean model_name for title (e.g. "gpt-4o" → "gpt4o")
        model_clean = model_name.replace('-', '').replace('-', '')

        # LLM-only bar (lighter + hatch)
        ax1.bar("LLM-only", llm_acc,
                color=lighter_colour, edgecolor=base_colour,
                hatch=hatch_pattern, linewidth=1.5)
        # BTE RAG bar (solid)
        ax1.bar("BTE RAG", rag_acc, color=base_colour, alpha=0.9)

        # labels and title at 12pt
        ax1.set_ylabel("Accuracy (%)", fontsize=12)
        ax1.set_title(model_clean, fontsize=12, pad=10)

        # tick labels at 12pt
        ax1.tick_params(axis='both', labelsize=12)

        ax1.set_ylim(0, 100)
        for x, v in enumerate([llm_acc, rag_acc]):
            ax1.text(x, v + 1, f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        sns.despine()
        fig1.tight_layout()
        _save(fig1, f"{col}_accuracy")
        plt.show()

        # ================= FIG 2 • Lift ==============================
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar("Lift", lift, color=base_colour)
        ax2.axhline(0, ls="--", lw=0.8, c="black")
        ax2.set_ylabel("Accuracy gain (pp)")
        ax2.set_title("RAG – LLM‑only", pad=8)
        ax2.set_ylim(min(0, lift) - 2, max(lift, 0) + 2)
        # ax2.text(0, lift + (2 if lift >= 0 else -2),
        #          f"{lift:+.2f}", ha="center",
        #          va="bottom" if lift >= 0 else "top", fontsize=10)

        ax2.text(0, lift + (3 if lift >= 0 else -3),
         f"{lift:+.2f}", ha="center", va="bottom",
         fontsize=11, weight="bold")
        ax2.set_title("RAG – LLM‑only", pad=20)
        sns.despine(); fig2.tight_layout(); _save(fig2, f"{col}_lift"); plt.show()

        # ================= FIG 3 • Counts (Incorrect/Correct) ========
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
        index = ["Incorrect", "Correct"]
        width = 0.35
        x_pos = range(len(index))
        # bars for LLM‑only
        ax3.bar([p - width/2 for p in x_pos], llm_cnt.values,
                width=width, color=lighter_colour,
                edgecolor=base_colour, hatch=hatch_pattern, label="LLM‑only")
        # bars for RAG
        ax3.bar([p + width/2 for p in x_pos], rag_cnt.values,
                width=width, color=base_colour, label="BTE RAG")
        ax3.set_xticks(x_pos); ax3.set_xticklabels(index)
        ax3.set_ylabel("Number of predictions")
        ax3.set_title("Prediction counts", pad=8)

        for rect in ax3.patches:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width() / 2,
                     height + 5,
                     f"{int(height)}",
                     ha="center", va="bottom", fontsize=8)

        ax3.set_ylim(0, max(llm_cnt.max(), rag_cnt.max()) * 1.15)
        ax3.legend(); sns.despine()
        fig3.tight_layout(); _save(fig3, f"{col}_counts"); plt.show()

        # ================= FIG 4 • Confusion matrix ==================
        cm = confusion_matrix(df[llm_col], df[col], labels=[True, False])

        fig4, ax4 = plt.subplots(figsize=(4.3, 3.8))
        
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    cmap=_seq_cmap(),
                    xticklabels=["RAG True", "RAG False"],
                    yticklabels=["LLM True", "LLM False"],
                    ax=ax4)
        ax4.set_xlabel("RAG prediction"); ax4.set_ylabel("LLM‑only reference")
        ax4.set_title("Confusion matrix", pad=10)
        sns.despine(left=False, bottom=False)
        fig4.tight_layout(); _save(fig4, f"{col}_confusion"); plt.show()

        # ---------------- summary row
        summary_rows.append({
            "rag_column"          : col,
            "llm_accuracy_%      ": round(llm_acc, 2),
            "rag_accuracy_%      ": round(rag_acc, 2),
            "accuracy_gain_pp    ": round(lift, 2),
            "relative_gain_%     ": round(lift / llm_acc * 100, 2) if llm_acc else None,
            "n_examples"         : n_total
        })

    return pd.DataFrame(summary_rows)


##############################################

def build_thresholded_context_df(
    questions_df,
    embedding_function,
    thresholds,
    min_threshold,
    context_volume,
    save_path
):
    # 1) load or initialize
    if os.path.exists(save_path):
        out_df = pd.read_csv(save_path)
    else:
        base_cols = [
            "id", "count", "question", "drug_name", "disease_name",
            "Drug_MeshID", "disease", "protein", "protein_name"
        ]
        thr_cols = []
        for thr in thresholds:
            thr_cols += [
                f"node_context_extracted_{thr}_compressed",
                f"combined_context_str_{thr}_compressed"
            ]
        out_df = pd.DataFrame(columns=base_cols + thr_cols)

    processed = set(out_df["question"])

    # 2) iterate questions
    for _, row in questions_df.iterrows():
        qid   = row["id"]
        question      = row["question"]
        drug_id       = row["Drug_MeshID"]
        drug_name     = row["drug_name"]
        disease_id    = row["disease"]
        disease_name  = row["disease_name"]
        count         = row["count"]
        protein  = row["protein_gene_symbol"]
        protein_name = row["protein_name"]

        if question in processed:
            continue

        # prepare row
        row_dict = {
            "id": qid,
            "count": count,
            "question": question,
            "drug_name": drug_name,
            "disease_name": disease_name,
            "Drug_MeshID": drug_id,
            "disease": disease_id,
            "protein": protein,
            "protein_name": protein_name,
        }

        # 3) loop thresholds
        for thr in thresholds:
            try:
                node_ctx, combined_ctx = retrieve_context_v17(
                    question, drug_id, drug_name,
                    disease_id, disease_name,
                    embedding_function,
                    thr,
                    min_threshold,
                    context_volume
                )
                # compress JSON strings
                nc_json = json.dumps(node_ctx)
                cc_json = json.dumps(combined_ctx)
                row_dict[f"node_context_extracted_{thr}_compressed"]   = zlib.compress(nc_json.encode("utf-8"))
                row_dict[f"combined_context_str_{thr}_compressed"]    = zlib.compress(cc_json.encode("utf-8"))

            except Exception as e:
                # on error, store None
                row_dict[f"node_context_extracted_{thr}_compressed"]   = None
                row_dict[f"combined_context_str_{thr}_compressed"]    = None
                print(f"retrieve_context_v17 error at threshold {thr}: {e}")

        # 4) append & save
        out_df = pd.concat([out_df, pd.DataFrame([row_dict])], ignore_index=True)
        out_df.to_csv(save_path, index=False)
        print(f"Built contexts for question {qid}")

    print("All done — contexts saved to", save_path)
    return out_df

######################## metabolite plots #################################

def _hist_panel(df, score_col, ax, y_max=200, label_offset=0.02):
    bins = np.arange(0, 1.05, 0.05)
    data = df[score_col].dropna()

    sns.histplot(
        data, bins=bins, kde=True,
        color='white', edgecolor='black',
        stat='count', ax=ax
    )

    ax.set_ylim(0, y_max)
    ax.set_xlim(0, 1)
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{b:.2f}' for b in bins], rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # bar labels
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                min(h + label_offset*y_max, y_max*0.98),
                f'{int(h)}',
                ha='center', va='bottom', fontsize=7
            )

    ax.set_xlabel('')   # remove per‑panel labels
    ax.set_ylabel('')


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

palette = sns.color_palette("colorblind", 2)
BLUE, ORANGE = palette[0], palette[1]

def darker(color, factor=0.5):
    r, g, b = to_rgb(color)
    return (r*factor, g*factor, b*factor)

DARK_BLUE   = darker(BLUE)
DARK_ORANGE = darker(ORANGE)

def make_similarity_figure(
    df, model_readable_name,
    sim_thresholds=range(10, 100, 10),
    include_full=True,
    cols_per_row=5,
    w_per_col=3.2, h_per_row=3.6,
    y_max=60,
    save_dir=None
):
        
    if save_dir:
        save_dir = Path(save_dir)
    else:
        save_dir = Path('.')      # current working dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ─ build columns & labels ───────────────────────────────────────────────
    score_cols = [f'cosine_similarity_sim_{t}' for t in sim_thresholds]
    if include_full:
        score_cols = ['cosine_similarity_NC'] + score_cols

    def nice(col):
        if col.endswith('_NC'):
            return 'Full context'
        pct = col.split('_')[-1]
        return f'Context lines > {pct}ᵗʰ percentile'

    col_labels = [nice(c) for c in score_cols]

    # ─ grid geometry ────────────────────────────────────────────────────────
    n_cols = min(cols_per_row, len(score_cols))
    n_rows = int(np.ceil(len(score_cols) / n_cols))
    fig_w, fig_h = w_per_col * n_cols, h_per_row * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharex='col', sharey='row',
        constrained_layout=False
    )
    axes = axes.flatten()

    # ─ plot each panel ──────────────────────────────────────────────────────
    for i, (col, label) in enumerate(zip(score_cols, col_labels)):
        ax = axes[i]
        _hist_panel(df, col, ax, y_max=y_max)

        # ▸ recolor bars: orange for “mini” or blue otherwise
        is_mini = 'mini' in model_readable_name.lower()
        base_c, dark_c = (ORANGE, DARK_ORANGE) if is_mini else (BLUE, DARK_BLUE)

        for patch in ax.patches:
            left, width = patch.get_x(), patch.get_width()
            center = left + width/2
            # dark-shade only for bins whose center ≥ 0.9
            patch.set_facecolor(dark_c if center >= 0.9 else base_c)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_title(label, fontsize=12, pad=6)

    # hide any unused axes
    for ax in axes[len(score_cols):]:
        ax.set_visible(False)

    # ─ layout tweaks ────────────────────────────────────────────────────────
    for ax in axes[:n_cols]:
        ax.tick_params(axis='x', labelbottom=True)

    fig.subplots_adjust(
        left=0.07, right=0.99,
        bottom=0.10, top=0.92,
        wspace=0.25, hspace=0.35
    )

    # ─ global title & labels ───────────────────────────────────────────────
    fig.suptitle(
        f'Distribution of Answer Similarities – {model_readable_name}',
        fontsize=12, fontweight='bold', y=1.0
    )
    fig.supxlabel(
        'Cosine similarity: model answer vs ground-truth answer',
        fontsize=12, y=0.0
    )
    fig.supylabel('No. of Questions', fontsize=12, x=0.02)

    # ─ save outputs ────────────────────────────────────────────────────────
    safe = model_readable_name.lower().replace(' ', '_').replace('-', '-')
    
    svg_path = save_dir / f'{safe}_similarity_distributions_hist.svg'
    fig.savefig(svg_path, format='svg')
    # fig.savefig(f'{safe}_similarity_distributions.png', dpi=300)

    plt.show()


palette = sns.color_palette("colorblind", 2)
BLUE, ORANGE = palette  # BLUE for GPT-4o, ORANGE for GPT-4o-mini

def darker(color, factor=0.5):
    """Return a darker shade of the given RGB color."""
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)

def plot_score_distribution(
    df,
    score_col='cosine_similarity',
    model_name='model',
    filename=None,
    fill_style='auto',  # 'auto', 'solid', or 'dashed'
    save_dir=None
):
    """
    Histogram with KDE and bar labels for cosine similarity scores.
    - Uses ORANGE for 'mini' models, BLUE otherwise.
    - Bars in the 0.9–1.0 range are darkened.
    - For 'dashed' fill_style (or auto→LLM-only), bars have no fill and dashed outline.
    - All text at fontsize=12.
    - Saves as SVG.
    """
    if save_dir:
        save_dir = Path(save_dir)
    else:
        save_dir = Path('.')      # current working dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data = df[score_col].dropna()
    bins = np.arange(0, 1.05, 0.05)

    # Base & dark colors
    base_color = ORANGE if 'mini' in model_name.lower() else BLUE
    dark_color = darker(base_color)

    # Decide hatch pattern
    name_l = model_name.lower()
    if fill_style == 'auto':
        hatch = '///' if 'llm-only' in name_l else None
    elif fill_style == 'dashed':
        hatch = '///'
    else:
        hatch = None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw histogram + KDE unchanged
    sns.histplot(
        data, bins=bins, kde=True,
        color=base_color, edgecolor='black',
        stat="count", ax=ax
    )

    # Recolor, hatch/outline and label bars
    for bar in ax.patches:
        left, width = bar.get_x(), bar.get_width()
        center = left + width / 2
        height = bar.get_height()

        if hatch:
            # LLM-only: no fill, dashed outline in base or dark color
            edge = dark_color if center >= 0.9 else base_color
            bar.set_facecolor('none')
            bar.set_edgecolor(edge)
            bar.set_linewidth(1)
            bar.set_hatch(hatch)
        else:
            # BTE-RAG or solid: filled bars as before
            bar.set_facecolor(dark_color if center >= 0.9 else base_color)
            bar.set_edgecolor('black')
            bar.set_linewidth(1)

        if height > 0:
            ax.text(
                center, height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=10
            )

    # Titles, labels, ticks & grid
    ax.set_title(f'Distribution of Cosine Similarity Scores: {model_name}', fontsize=20)
    ax.set_xlabel('Cosine Similarity', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.set_xticks(bins)
    ax.set_ylim(top=50)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save
    if filename is None:
        safe = model_name.lower().replace(' ', '_').replace('-', '_')
        filename = save_dir / f'{safe}_similarity_distribution.svg'
    fig.savefig(filename, format='svg')
    plt.show()




