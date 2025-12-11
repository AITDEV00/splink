import requests
import json

def fetch_and_stream_to_list():
    base_url = "http://localhost:8000/graph/entity/list"
    output_file = "all_entities.json"
    limit = 1000
    offset = 0
    
    headers = {'accept': 'application/json', 'x-workspace': 'maalmarri'}
    
    # Flag to track if we need to add a comma before the object
    is_first_item = True

    print(f"Streaming download to '{output_file}'...")

    with open(output_file, "w", encoding="utf-8") as f:
        # 1. Start the JSON list
        f.write('[')
        
        while True:
            params = {"limit": limit, "offset": offset}

            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                
                batch = response.json()

                # STOP CONDITION: No more data
                if not batch:
                    break

                # 2. Write items one by one (or batch by batch)
                for entity in batch:
                    if not is_first_item:
                        f.write(',\n') # Add comma if it's not the first item
                    
                    # Write the entity object
                    json.dump(entity, f, ensure_ascii=False)
                    is_first_item = False

                print(f"Saved batch at offset {offset}. (Items in this batch: {len(batch)})")
                offset += limit

            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                break
        
        # 3. Close the JSON list
        f.write(']')

    print("Done. File is closed and valid.")

if __name__ == "__main__":
    fetch_and_stream_to_list()