import json
import os
import sys

# --- Configuration ---
OUTPUT_FILE = "webnlg_graph_qa_val.json"

def sort_json_by_id(filename):
    """
    Loads a JSON file containing a list of objects, sorts the list 
    based on the 'id' key of each object, and saves it back.
    """
    print(f"--- Starting sorting process for '{filename}' ---")
    
    # 1. Check if the file exists and is not empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"Error: The file '{filename}' was not found or is empty. Cannot sort.")
        return

    # 2. Load the data
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{filename}' as a JSON array. Please check the file's integrity.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    # Ensure the loaded data is a list
    if not isinstance(data, list):
        print(f"Error: Data in '{filename}' is not a list/array. Sorting skipped.")
        return

    original_length = len(data)
    print(f"Successfully loaded {original_length} records.")

    # 3. Sort the list
    try:
        # The key=lambda x: x['id'] tells Python to use the value of the 'id' field 
        # for sorting, which is the fastest way to do this.
        sorted_data = sorted(data, key=lambda x: x.get('id', sys.maxsize)) 
        
        # We use x.get('id', sys.maxsize) to handle cases where an item might 
        # somehow be missing the 'id' key by placing them at the end.

    except TypeError as e:
        print(f"Error during sorting: {e}. Ensure all items have a numerical 'id' field.")
        return
    
    # 4. Save the sorted data back to the file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Sorting complete! {original_length} records have been sorted by 'id' and saved to '{filename}'.")
        
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    sort_json_by_id(OUTPUT_FILE)