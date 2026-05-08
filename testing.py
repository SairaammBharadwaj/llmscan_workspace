import os

def scout_all_files(root_folder="data", num_lines=5, max_chars=150):
    print(f"🔍 Starting Data X-Ray in folder: '{root_folder}'\n")
    
    if not os.path.exists(root_folder):
        print(f"❌ Error: The folder '{root_folder}' does not exist.")
        # Sometimes unzipping creates a nested folder, let's try 'data/data'
        if os.path.exists("data/data"):
            print("Found 'data/data', switching to that directory...\n")
            root_folder = "data/data"
        else:
            return

    file_count = 0
    
    # os.walk goes through the main folder and all sub-folders automatically
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # We only want to peek inside text-based files
            if filename.endswith(('.json', '.csv', '.txt', '.jsonl')):
                file_count += 1
                filepath = os.path.join(dirpath, filename)
                
                print(f"📂 {filepath}")
                print("-" * 60)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for i in range(num_lines):
                            line = f.readline()
                            if not line: # Stop if the file is shorter than 5 lines
                                break
                            
                            # Clean up the line and limit its length so it doesn't flood the screen
                            clean_line = line.strip()
                            if len(clean_line) > max_chars:
                                clean_line = clean_line[:max_chars] + " ... [TRUNCATED]"
                                
                            print(f"  Line {i+1}: {clean_line}")
                except Exception as e:
                    print(f"  [Error reading file: {e}]")
                
                print("\n") # Add space between files

    if file_count == 0:
        print("No readable dataset files found.")
    else:
        print(f"✅ X-Ray complete. Scanned {file_count} files.")

if __name__ == "__main__":
    scout_all_files()