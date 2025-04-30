"""Main application module."""

#from .analysis.analyzeface import process_images
from config.settings import CHATLOGS_DIR, PILOT_CSV
from preprocessing.cleaner import process_single_chatlog
 #process_chatlogs, create_directories

if __name__ == "__main__":
    # create directories and save raw chatlogs
    #create_directories(PILOT_CSV, CHATLOGS_DIR) 
    # process the chatlogs
    #process_chatlogs(CHATLOGS_DIR)

    process_single_chatlog("chatlogsM/24/raw_chatlog.txt")
    # Finally process images
   # process_images(IMAGES_DIR, IMAGES_OUTPUT_CSV) 
