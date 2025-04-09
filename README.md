# Image Processor

This script automatically processes images by:
1. Watching a designated folder for new images
2. Renaming them to "foobar" with the original extension
3. Allowing you to manually specify which numbered directory to move them to

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the script:
   ```
   python image_processor.py
   ```

## How It Works

1. The script creates a folder called `drop_folder` in the same directory
2. When you drag and drop an image into the `drop_folder`, the script will:
   - Detect the new image
   - Prompt you to enter which numbered directory to move it to
   - Rename it to `foobar.[extension]`
   - Move it to the specified directory in the `images` folder

## Configuration

You can modify the following settings in the script:
- `WATCH_FOLDER`: The folder to watch for new images (default: "drop_folder")
- `IMAGES_BASE_DIR`: The base directory for numbered folders (default: "images")
- `IMAGE_PREFIX`: The prefix for renamed images (default: "foobar")
- `IMAGE_EXTENSIONS`: Supported image file extensions

## Stopping the Script

Press `Ctrl+C` in the terminal to stop the script. 