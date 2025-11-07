import json

# Replace 'input_file.json' with the actual name of your file
file_name = 'webnlg_graph_qa_val.json' 

try:
    with open(file_name, 'r') as f:
        data = json.load(f)

    # 1. Extract all 'id' values
    present_ids = {item.get('id') for item in data if isinstance(item, dict) and 'id' in item}

    # 2. Define the expected range (0 to 35425)
    # range(35426) generates numbers from 0 up to 35425
    expected_ids = set(range(1667)) 

    # 3. Find the missing IDs
    missing_ids = sorted(list(expected_ids - present_ids))

    # Print a summary
    print(f"Total expected IDs: {len(expected_ids)}")
    print(f"Total present IDs: {len(present_ids)}")
    print(f"Total missing IDs found: {len(missing_ids)}")

    # Display first and last few missing IDs
    print(f"Missing IDs (first 20): {missing_ids[:20]}")
    print(f"Missing IDs (last 20): {missing_ids[-20:]}")

    # Save the complete list of missing IDs to a file
    with open('missing_ids.txt', 'w') as f_out:
        f_out.write('\n'.join(map(str, missing_ids)))

    print("The complete list of missing IDs has been saved to 'missing_ids.txt'.")

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode '{file_name}' as a JSON array.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")