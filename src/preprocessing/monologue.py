import os 

def create_monologue():
    # Base directories
    chatlogs_dir = "chatlogsD"
    monologue_dir = "chatlogsM"
    
    # Process directories 0-40
    for i in range(41):
        # Source path for cleaned chatlog
        source_path = os.path.join(chatlogs_dir, str(i), "cleaned_chatlog.txt")
        
        # Skip if source file doesn't exist
        if not os.path.exists(source_path):
            print(f"Skipping directory {i}: cleaned_chatlog.txt not found")
            continue
        
        # Read and process the chatlog
        with open(source_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out ChatGPT blocks and keep only "You said:" blocks
        monologue_lines = []
        skip_mode = False
        
        for line in lines:
            if line.startswith("ChatGPT said:"):
                skip_mode = True
            elif line.startswith("You said:"):
                skip_mode = False
                monologue_lines.append(line)
            elif not skip_mode:
                monologue_lines.append(line)
        
        # Write to new file
        output_path = os.path.join(monologue_dir, str(i), "cleaned_monologue.txt")
        with open(output_path, 'w') as f:
            f.writelines(monologue_lines)
            
        print(f"Processed directory {i}: created cleaned_monologue.txt")

def delete_files():
    # Base directory
    monologue_dir = "chatlogsM"
    
    # Process directories 0-40
    for i in range(41):
        # Path to raw_monologue.txt
        file_path = os.path.join(monologue_dir, str(i), "raw_monologue.txt")
        
        # Delete if exists
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted raw_monologue.txt from directory {i}")
        else:
            print(f"No raw_monologue.txt found in directory {i}")

   