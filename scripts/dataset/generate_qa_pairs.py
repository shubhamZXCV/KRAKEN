import time
import json
import os
import sys
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

# Gemini 2.5 Flash-Lite Rate Limits
REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 1000
DELAY_BETWEEN_REQUESTS = (60 / REQUESTS_PER_MINUTE) + 0.1 # 4.1 seconds
OUTPUT_FILE = "webnlg_graph_qa_val.json" 
DAILY_LOG_FILE = "daily_progress_log.txt"
DATASET_NAME = "GEM/web_nlg"
DATASET_CONFIG = "en"  
DATASET_SPLIT = "validation" 

# --- Few-Shot Prompt Template (Kept the same) ---

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

# --- Data Handling Functions (Kept the same) ---

def load_data():
    """
    Loads the GEM/web_nlg dataset (English, train split) and extracts triples and sentences.
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

def get_last_processed_index():
    """Reads the last successful index from the daily log."""
    try:
        with open(DAILY_LOG_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                return int(lines[-1].strip().split(":")[1].split(',')[0].strip())
            return 0
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"Error reading log file: {e}. Starting from index 0.")
        return 0

def log_progress(index, requests_processed_today, current_key_index):
    """Logs the last processed index, daily count, and current key index."""
    with open(DAILY_LOG_FILE, 'a') as f:
        f.write(f"Index:{index}, Daily_Count:{requests_processed_today}, Key_Index:{current_key_index}, Time:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# --- Key Management Functions ---

def load_current_key_index():
    """Loads the last used key index from a separate file."""
    try:
        with open(KEY_INDEX_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0 # Start with the first key if file doesn't exist or is empty

def save_current_key_index(index):
    """Saves the index of the currently active key."""
    with open(KEY_INDEX_FILE, 'w') as f:
        f.write(str(index))

# --- API Call Function (Kept the same) ---

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


def run_generation_job():
    """Main execution loop that handles key switching, rate limits, and saving."""
    
    if not API_KEY_LIST or API_KEY_LIST[0] == "YOUR_SECOND_GEMINI_API_KEY_HERE":
        print("FATAL ERROR: Please populate the API_KEY_LIST with valid keys.")
        sys.exit(1)
        
    # --- Initialization ---
    
    data = load_data()
    total_data_size = len(data)
    results = load_or_initialize_results(OUTPUT_FILE)
    
    # Resumption logic
    start_index_json = len(results)
    start_index_log = get_last_processed_index()
    start_index = max(start_index_json, start_index_log)
    
    current_key_index = load_current_key_index()
    requests_processed_today = 0
    
    print(f"\n--- Starting Multi-QA Generation Job ---")
    print(f"Resuming from WebNLG entry index: {start_index}")
    print(f"Starting with API Key #{current_key_index + 1}/{len(API_KEY_LIST)}")
    print(f"Daily limit per key: {REQUESTS_PER_DAY} RPD")
    print("-" * 30)

    # --- Client Creation ---
    
    # Get the current key and initialize the client
    current_api_key = API_KEY_LIST[current_key_index]
    client = genai.Client(api_key=current_api_key)


    for i in range(start_index, total_data_size):
        # --- RPD Check: Switch Key if Limit Reached ---
        if requests_processed_today >= REQUESTS_PER_DAY:
            print("\n*** RPD LIMIT REACHED for current key ***")
            current_key_index += 1
            requests_processed_today = 0  # Reset daily counter
            
            if current_key_index >= len(API_KEY_LIST):
                print("All API keys have reached their daily limit. Stopping job.")
                break 
                
            # Switch to the new key
            current_api_key = API_KEY_LIST[current_key_index]
            client = genai.Client(api_key=current_api_key)
            save_current_key_index(current_key_index)
            print(f"Switched to API Key #{current_key_index + 1}")
            time.sleep(5) # Brief pause after switching keys

        entry = data[i]
        
        try:
            qa_pairs_list = generate_qa_pair(client, entry)
            
            output_entry = {
                "id": i,
                "webnlg_id": entry['webnlg_id'],
                "triples": entry['triples'],
                "sentence": entry['sentence'],
                "qa_pairs": qa_pairs_list,
            }

            # 1. Append to the list in memory
            results.append(output_entry)
            
            # 2. Update counters and log
            requests_processed_today += 1
            # Log the current key index with the progress
            log_progress(i + 1, requests_processed_today, current_key_index) 
            
            # 3. CRITICAL: Save the entire list to the JSON file immediately
            save_results(OUTPUT_FILE, results)
            
            num_generated = len(qa_pairs_list)
            key_display = current_key_index + 1
            print(f"[{i+1}/{total_data_size} | Key #{key_display}, Day Count: {requests_processed_today}] Success! Generated {num_generated} Q/A pairs. Saved to disk.")
            
        except ResourceExhausted:
            # If the API returns a 429 error (rate limit), it means the key is exhausted (either RPM or TPM).
            print("\n*** API RATE LIMIT ERROR (RPM/TPM) ***")
            print("Pausing for 60 seconds to reset minute-based limit.")
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
    run_generation_job()