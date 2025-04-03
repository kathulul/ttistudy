import csv 
import os  # Add this import at the top

def remove_gpt(cleaned_chatlog):
    pass

def clean_chatlog(chatlog:str):
    # do all your cleaning processes in here
    pass

def process_chatlogs(p_csv:str):
    # Load CSV file into a dictionary with response IDs as keys
    participant_info = {}
    with open(p_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        i=0
        
        for row in csv_reader:
            if i < 2:
                i+=1
                continue
            response_id = row['ResponseId']
            participant_info[response_id] = {
                'Gender': row['Gender'],
                'Race': row['Race'],
                'Chatlog': row['Chatlog']
            }
            # Create directory for this participant if it doesn't exist
            pid_dir = os.path.join('chatlogs', str(i-2))
            os.makedirs(pid_dir, exist_ok=True)
            
            # Save raw chatlog to a file
            chatlog_path = os.path.join(pid_dir, 'raw_chatlog.txt')
            with open(chatlog_path, 'w') as chatlog_file:
                chatlog_file.write(row['Chatlog'])
            i+=1

    cleaned_chatlog = clean_chatlog(chatlog)
    dialogue_removed_log = remove_gpt(cleaned_chatlog)
    # save chatlog, cleaned_log, dialogue removed to each file probably under like chatlogs/pid/
    # Add participant info , chatlog, cleaned_chatlog, dialogue_removed_log
    pass

if __name__ == "__main__":
    process_chatlogs("data/testdata.csv")

