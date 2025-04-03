import csv 
import os  

def remove_gpt(cleaned_chatlog):
    pass

def clean_chatlog(chatlog:str):

    return ""  # Placeholder for the cleaned chatlog

def process_chatlogs(p_csv:str):
    # Load CSV file into a dictionary with response IDs as keys
    participant_info = {}
    with open(p_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        i = 0
        
        for row in csv_reader:
            if i < 2:
                i += 1
                continue
            response_id = row['ResponseId']
            participant_info[response_id] = {
                'Gender': row['Gender'],
                'Race': row['Race'],
                'Chatlog': row['Chatlog']
            }
            # Create directory for this participant if it doesn't exist
            pid_dir = os.path.join('chatlogs', str(i - 2))
            os.makedirs(pid_dir, exist_ok=True)
            
            # Save raw chatlog to a file
            chatlog_path = os.path.join(pid_dir, 'raw_chatlog.txt')
            with open(chatlog_path, 'w') as chatlog_file:
                chatlog_file.write(row['Chatlog'])
            i += 1

    # Loop over each PID directory and clean chatlogs
    for pid in os.listdir('chatlogs'):
        pid_path = os.path.join('chatlogs', pid)
        if os.path.isdir(pid_path):
            chatlog_file_path = os.path.join(pid_path, 'raw_chatlog.txt')
            if os.path.exists(chatlog_file_path):
                with open(chatlog_file_path, 'r') as chatlog_file:
                    chatlog_content = chatlog_file.read()
                    cleaned_chatlog = clean_chatlog(chatlog_content)
                    
                    # Save cleaned chatlog to a new file
                    cleaned_chatlog_path = os.path.join(pid_path, 'cleaned_chatlog.txt')
                    with open(cleaned_chatlog_path, 'w') as cleaned_file:
                        cleaned_file.write(cleaned_chatlog)

if __name__ == "__main__":
    process_chatlogs("data/testdata.csv")

