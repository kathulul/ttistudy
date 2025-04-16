import csv
import os


""" this script is used to create image directories for each participant only"""


def process_chatlogs(p_csv:str):
    # First loop: Process the CSV and create directories
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
            pid_dir = os.path.join('images', str(i - 2))
            os.makedirs(pid_dir, exist_ok=True)
            i += 1
    
    return participant_info

if __name__ == "__main__":
    process_chatlogs("data/Chatbot Pilot_April 9, 2025_14.43.csv")

