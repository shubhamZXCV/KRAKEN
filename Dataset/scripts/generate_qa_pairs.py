import time
import json
import os
import sys
# Corrected Imports for robust error handling
from google import genai
from google.genai import types
from google.genai.errors import APIError 
from google.api_core.exceptions import ResourceExhausted # Correct exception for 429/rate limits
from datasets import load_dataset # For loading WebNLG

# --- Configuration and Limits ---

# IMPORTANT: I've kept your provided key for context, but in a real scenario, 
# you should NEVER share your API key publicly.
API_KEY = "" 

# Gemini 2.5 Flash-Lite Rate Limits
REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 1000
DELAY_BETWEEN_REQUESTS = (60 / REQUESTS_PER_MINUTE) + 0.1 # 4.1 seconds to stay safe
OUTPUT_FILE = "webnlg_graph_qa_dataset.json" 
DAILY_LOG_FILE = "daily_progress_log.txt"
DATASET_NAME = "GEM/web_nlg"
DATASET_CONFIG = "en"  # English language
DATASET_SPLIT = "train" # Use the train split

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

# --- Data Handling Functions ---

def load_data():
    """
    Loads the GEM/web_nlg dataset (English, train split) and extracts triples and sentences.
    """
    print(f"--- Loading {DATASET_NAME}/{DATASET_CONFIG} split: {DATASET_SPLIT} ---")
    
    try:
        # Load the English training split of the dataset
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run: pip install datasets")
        sys.exit(1) # Exit if data can't be loaded

    # Map the dataset columns to the required format
    webnlg_data = []
    for entry in dataset:
        # 'input' is the list of triples (e.g., ["S | P | O"])
        # 'target' is the single natural language sentence
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
    # This function is now called frequently, ensuring data persists.
    with open(filename, 'w', encoding='utf-8') as f:
        # Using a small indent (2) for slightly better performance while maintaining readability
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- Rate Limit & Progress Functions ---

def get_last_processed_index():
    """Reads the last successful index from the daily log."""
    try:
        with open(DAILY_LOG_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                # The index is stored as 'Index:<value>'
                return int(lines[-1].strip().split(":")[1].split(',')[0].strip())
            return 0
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"Error reading log file: {e}. Starting from index 0.")
        return 0

def log_progress(index, requests_processed_today):
    """Logs the last processed index and the daily count."""
    with open(DAILY_LOG_FILE, 'a') as f:
        f.write(f"Index:{index}, Daily_Count:{requests_processed_today}, Time:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def generate_qa_pair(client, entry):
    """Calls the Gemini API to generate the list of QA pairs."""
    
    triples_str = " ; ".join(entry['triples'])
    num_triples = len(entry['triples'])
    
    prompt = PROMPT_TEMPLATE.format(
        num_triples=num_triples,
        triples_str=triples_str, 
        sentence=entry['sentence']
    )
    
    # Define the required response structure: a list of QA objects
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
    """Main execution loop that handles rate limits and simultaneous saving."""
    
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("FATAL ERROR: Please set your Gemini API key in the script.")
        sys.exit(1)
        
    client = genai.Client(api_key=API_KEY)
    data = load_data()
    total_data_size = len(data)
    
    # Load existing results and determine the starting index
    results = load_or_initialize_results(OUTPUT_FILE)
    
    # Determine the actual starting index based on either the JSON file size or the log file
    start_index_json = len(results)
    start_index_log = get_last_processed_index()
    start_index = max(start_index_json, start_index_log)
    
    requests_processed_today = 0
    
    print(f"\n--- Starting Multi-QA Generation Job ---")
    print(f"Resuming from WebNLG entry index: {start_index} (JSON count: {start_index_json}, Log index: {start_index_log})")
    print(f"Daily limit (API Calls): {REQUESTS_PER_DAY} RPD")
    print("-" * 30)

    for i in range(start_index, total_data_size):
        if requests_processed_today >= REQUESTS_PER_DAY:
            print("\n*** DAILY LIMIT REACHED ***")
            print(f"Processed {requests_processed_today} API calls today. Waiting for 24 hours.")
            break 

        entry = data[i]
        
        try:
            qa_pairs_list = generate_qa_pair(client, entry)
            
            # Create a single output entry
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
            log_progress(i + 1, requests_processed_today)
            
            # 3. CRITICAL: Save the entire list to the JSON file immediately
            save_results(OUTPUT_FILE, results)
            
            num_generated = len(qa_pairs_list)
            print(f"[{i+1}/{total_data_size} | Day Count: {requests_processed_today}] Success! Generated {num_generated} Q/A pairs. Saved to disk.")
            
        except ResourceExhausted: # Correctly catching the 429 error
            print("\n*** API RATE LIMIT ERROR (RPM or TPM) ***")
            print(f"Pausing for 60 seconds to reset the minute-based limit. Index: {i}")
            time.sleep(60) 
            continue # Try the current index again

        except APIError as e:
            print(f"\n[Error at Index {i}] An API error occurred: {e}")
            print("Skipping this entry and continuing...")
            continue 

        except Exception as e:
            # Catching JSONDecodeError or other unexpected errors
            print(f"\n[Error at Index {i}] An unexpected error occurred: {e}. Output may be malformed.")
            print("Skipping this entry and continuing...")
            continue 

        # --- Enforce RPM Limit ---
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nJob finished for the day/dataset completion. Total entries saved to '{OUTPUT_FILE}': {len(results)}")


if __name__ == "__main__":
    run_generation_job()