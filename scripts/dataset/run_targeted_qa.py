import time
import json
import os
import sys
# NOTE: Using the standard import for the Gemini API
from google import genai
from google.genai import types
from google.genai.errors import APIError 
from google.api_core.exceptions import ResourceExhausted 
from datasets import load_dataset 

# --- Configuration and Limits ---

# 🛑 CRITICAL: REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL API KEYS 🛑
# The script will use the next key in the list once 1000 requests are completed.
API_KEY_LIST = [
    "AIzaSyB9K2sxcDqJEqsbnnD2PZbcUtkhVy3c51o",  # Key 1 (Your current key)
    "AIzaSyDxEKT8ShIDjAUWWQzu0PyFnjgz2WWJF54",
    "AIzaSyAS2zqeMp5y2Qyyr-idVaA7BvGE2Yt5fL0",
    "AIzaSyC3D4SJNEGQK2IQnX1BY1DsnRmcwSqxdPQ",
    "AIzaSyD3hI8NoQvdA-p-8O8se1xZP1J9NMtvaeY",
    "AIzaSyAQ1b7Q0LG_t91Ws4ZCOsiA1_UkJ401-FY",
    "AIzaSyBdT0qAtN9o2RPkVa2GnZk-x-9kx1XU9vo",
    "AIzaSyBMD4B79Iegyydw0El8Z_a_c_xZwIFcagQ"
    # Add as many keys as you need for total throughput.
]
# We will use the index of the key in this list to track progress across runs.
KEY_INDEX_FILE = "api_key_index.txt" 
OUTPUT_FILE = "webnlg_graph_qa_val.json" 
DAILY_LOG_FILE = "daily_progress_log.txt"
MISSING_IDS_FILE = "missing_ids.txt" # <--- NEW: File containing IDs to process

# Gemini 2.5 Flash-Lite Rate Limits
REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 1000 # Daily limit per key
DELAY_BETWEEN_REQUESTS = (60 / REQUESTS_PER_MINUTE) + 0.1 

# Dataset Configuration
DATASET_NAME = "GEM/web_nlg"
DATASET_CONFIG = "en"  
DATASET_SPLIT = "validation" 

# --- Few-Shot Prompt Template ---

PROMPT_TEMPLATE = """
You are an expert Question-Answer pair generator for a Graph QA task. Your goal is to generate multiple, distinct, and precise question-answer pairs for the given <TRIPLES> and <SENTENCE>.

The number of triples is {num_triples}.

Constraints:
1. Generate between 1 and 4 Q/A pairs. For 1-2 triples, generate 1-2 pairs. For 3+ triples, generate 3-4 distinct pairs.
2. Each question must target a specific entity (subject or object) or relationship (predicate) from the <TRIPLES>.
3. Each question must be answerable using only the information in the sentence.
4. The answer must be a short, exact phrase or entity.
5. Output ONLY a single JSON list containing multiple question-answer objects. Do not include any other text or markdown formatting.

---
Example 1: (2 triples)
<TRIPLES>: U2 | agent | Island Records ; U2 | genre | Rock
<SENTENCE>: U2 is an Irish rock band whose agent is Island Records.
<OUTPUT_JSON>: [
    {{ "question": "What is the music genre of U2?", "answer": "Rock" }},
    {{ "question": "Which record label acts as the agent for U2?", "answer": "Island Records" }}
]
---
Example 2: (4 triples)
<TRIPLES>: France | capital | Paris ; France | language | French ; Paris | population | 2.141 Million ; Paris | country | France
<SENTENCE>: Paris is the capital city of France, where French is the official language. Paris has a population of 2.141 million.
<OUTPUT_JSON>: [
    {{ "question": "Which city is the capital of France?", "answer": "Paris" }},
    {{ "question": "What is the official language spoken in France?", "answer": "French" }},
    {{ "question": "What is the population of the capital city Paris?", "answer": "2.141 million" }},
    {{ "question": "In which country is the city of Paris located?", "answer": "France" }}
]
---
Now generate the pairs:
<TRIPLES>: {triples_str}
<SENTENCE>: {sentence}
<OUTPUT_JSON>:
"""

# --- Data Handling and Utility Functions ---

def load_data():
    """
    Loads the GEM/web_nlg dataset (English, train split).
    """
    print(f"--- Loading {DATASET_NAME}/{DATASET_CONFIG} split: {DATASET_SPLIT} ---")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run: pip install datasets")
        sys.exit(1) 

    webnlg_data = []
    for entry in dataset:
        webnlg_data.append({
            "triples": entry['input'],
            "sentence": entry['target'],
            "webnlg_id": entry.get('webnlg_id', None) 
        })
    
    print(f"Loaded {len(webnlg_data)} data points from the dataset.")
    return webnlg_data

def load_or_initialize_results(filename):
    """Loads existing results from the JSON file or returns an empty list."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing JSON file '{filename}'. Starting fresh.")
            return []
    return []

def save_results(filename, data):
    """Writes the entire list of results back to the single JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log_progress(index, requests_processed_today, current_key_index):
    """Logs the last processed index, daily count, and current key index."""
    with open(DAILY_LOG_FILE, 'a') as f:
        f.write(f"Index:{index}, Daily_Count:{requests_processed_today}, Key_Index:{current_key_index}, Time:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def load_current_key_index():
    """Loads the last used key index from a separate file."""
    try:
        with open(KEY_INDEX_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0 

def save_current_key_index(index):
    """Saves the index of the currently active key."""
    with open(KEY_INDEX_FILE, 'w') as f:
        f.write(str(index))

def load_missing_ids_from_file(filename):
    """
    Loads the list of missing IDs from the specified text file.
    Assumes one integer ID per line.
    """
    try:
        with open(filename, 'r') as f:
            ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        print(f"Successfully loaded {len(ids)} missing IDs from '{filename}'.")
        return ids
    except FileNotFoundError:
        print(f"FATAL ERROR: Missing IDs file '{filename}' not found. Cannot proceed with targeted run.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Failed to read or parse IDs from '{filename}': {e}")
        sys.exit(1)


# --- API Call Function (Unchanged) ---

def generate_qa_pair(client, entry):
    """Calls the Gemini API to generate the list of QA pairs."""
    
    triples_str = " ; ".join(entry['triples'])
    num_triples = len(entry['triples'])
    
    prompt = PROMPT_TEMPLATE.format(
        num_triples=num_triples,
        triples_str=triples_str, 
        sentence=entry['sentence']
    )
    
    qa_pair_schema = {
        "type": "object", 
        "properties": {
            "question": {"type": "string"}, 
            "answer": {"type": "string"}
        }
    }
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={"type": "array", "items": qa_pair_schema}
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config=config,
    )
    
    return json.loads(response.text)


# --- Main Execution Loop ---

def run_targeted_generation_job():
    """
    Main execution loop that loads the dataset, uses the list of IDs from 
    missing_ids.txt, and targets generation only for those IDs.
    """
    
    # if not API_KEY_LIST or API_KEY_LIST[0] == "AIzaSyB9K2sxcDqJEqsbnnD2PZbcUtkhVy3c51o":
    #     print("FATAL ERROR: Please populate the API_KEY_LIST with valid, personal keys.")
    #     return
        
    # --- Initialization ---
    
    data = load_data()
    
    # 1. Load the specific list of IDs to process from the text file
    all_target_indices = load_missing_ids_from_file(MISSING_IDS_FILE)
    
    # 2. Load existing results
    results = load_or_initialize_results(OUTPUT_FILE)
    
    # 3. Filter the target indices against what's already in the JSON file 
    # (in case the missing_ids.txt is slightly out of date)
    processed_ids_in_json = {item.get('id') for item in results if isinstance(item, dict) and 'id' in item}
    indices_to_process = sorted([i for i in all_target_indices if i not in processed_ids_in_json])
    
    if not indices_to_process:
        print("All IDs listed in 'missing_ids.txt' are already present in the output file. Job complete.")
        return

    total_jobs_to_run = len(indices_to_process)
    
    # --- Key Management ---
    current_key_index = load_current_key_index()
    requests_processed_today = 0
    
    print(f"\n--- Starting TARGETED Multi-QA Generation Job ---")
    print(f"Targeting {total_jobs_to_run} entries.")
    print(f"Starting with API Key #{current_key_index + 1}/{len(API_KEY_LIST)}")
    print(f"Daily limit per key: {REQUESTS_PER_DAY} RPD")
    print("-" * 30)

    # --- Client Creation ---
    current_api_key = API_KEY_LIST[current_key_index]
    try:
        client = genai.Client(api_key=current_api_key)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Gemini client: {e}")
        return

    # Iterate over the specific original indices that are missing
    for job_count, i in enumerate(indices_to_process):
        # --- RPD Check: Switch Key if Limit Reached ---
        if requests_processed_today >= REQUESTS_PER_DAY:
            print("\n*** RPD LIMIT REACHED for current key ***")
            current_key_index += 1
            requests_processed_today = 0
            
            if current_key_index >= len(API_KEY_LIST):
                print("All API keys have reached their daily limit. Stopping job.")
                break 
                
            current_api_key = API_KEY_LIST[current_key_index]
            client = genai.Client(api_key=current_api_key)
            save_current_key_index(current_key_index)
            print(f"Switched to API Key #{current_key_index + 1}")
            time.sleep(5) 

        # Ensure index is within dataset bounds before accessing
        if i >= len(data):
            print(f"Warning: Index {i} found in missing_ids.txt exceeds dataset size ({len(data)}). Skipping.")
            continue
            
        entry = data[i] # Get the correct entry from the main dataset using the missing index 'i'
        
        try:
            qa_pairs_list = generate_qa_pair(client, entry)
            
            output_entry = {
                "id": i, # CRITICAL: Use the original index 'i' as the ID
                "webnlg_id": entry['webnlg_id'],
                "triples": entry['triples'],
                "sentence": entry['sentence'],
                "qa_pairs": qa_pairs_list,
            }

            # 1. Append the new data to the list in memory
            results.append(output_entry)
            
            # 2. Update counters and log
            requests_processed_today += 1
            log_progress(i, requests_processed_today, current_key_index) 
            
            # 3. CRITICAL: Save the entire (updated) list back to the JSON file
            save_results(OUTPUT_FILE, results)
            
            num_generated = len(qa_pairs_list)
            key_display = current_key_index + 1
            print(f"[JOB {job_count+1}/{total_jobs_to_run} | Index: {i} | Key #{key_display}, Day Count: {requests_processed_today}] Success! Generated {num_generated} Q/A pairs. Saved to disk.")
            
        except ResourceExhausted:
            print("\n*** API RATE LIMIT ERROR (RPM/TPM) ***")
            print("Pausing for 60 seconds to reset limit.")
            time.sleep(60) 
            continue 
        except APIError as e:
            print(f"\n[Error at Index {i}] An API error occurred: {e}")
            print("Skipping this entry and continuing...")
            continue 
        except Exception as e:
            print(f"\n[Error at Index {i}] An unexpected error occurred: {e}. Skipping...")
            continue 

        # --- Enforce RPM Limit ---
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nJob finished. Total entries saved to '{OUTPUT_FILE}': {len(results)}")


if __name__ == "__main__":
    run_targeted_generation_job()