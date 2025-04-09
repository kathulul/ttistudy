import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re

class ImageHandler(FileSystemEventHandler):
    def __init__(self):
        self.image_pattern = re.compile(r'^(\d{1,2})\.(png|jpe?g)$', re.IGNORECASE)
        
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)
            
    def process_file(self, file_path):
        filename = os.path.basename(file_path)
        match = self.image_pattern.match(filename)
        
        if match:
            number = match.group(1)
            extension = match.group(2)
            
            # Ensure the number is between 0-40
            if 0 <= int(number) <= 40:
                # Create target directory if it doesn't exist
                target_dir = os.path.join('images', number)
                os.makedirs(target_dir, exist_ok=True)
                
                # New filename
                new_filename = f'4o_DB.{extension}'
                target_path = os.path.join(target_dir, new_filename)
                
                # Move and rename the file
                try:
                    shutil.move(file_path, target_path)
                    print(f"Moved {filename} to {target_path}")
                except Exception as e:
                    print(f"Error moving file {filename}: {e}")

def main():
    # Create the images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Set up the observer
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path='drop_folder', recursive=False)
    observer.start()
    
    print("Watching drop_folder for new images...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
