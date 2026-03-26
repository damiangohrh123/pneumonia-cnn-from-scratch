import os
import json
from PIL import Image  # Only used for the initial conversion
from typing import List

def process_directory(source_dir: str, target_dir: str, size: int = 64) -> None:
    """
    Converts JPEG images into normalized JSON lists of lists.
    
    Args:
        source_dir: Path to the Kaggle folder (e.g., 'data/train/PNEUMONIA').
        target_dir: Path to save the JSON files.
        size: The target dimensions (64x64).
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Check if source directory exists to avoid crashes
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} not found. Skipping...")
        return

    for filename in os.listdir(source_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                # 1. Open and convert to Grayscale ('L' mode)
                with Image.open(os.path.join(source_dir, filename)) as img:
                    img = img.convert('L')
                    
                    # 2. Resize to 64x64
                    img = img.resize((size, size))
                    
                    # 3. Convert to a Python List of Lists
                    pixels: List[List[int]] = []
                    for y in range(size):
                        row = [img.getpixel((x, y)) for x in range(size)]
                        pixels.append(row)
                    
                    # 4. Save as JSON
                    # rsplt ensures we handle filenames with multiple dots correctly
                    json_name = filename.rsplit('.', 1)[0] + ".json"
                    with open(os.path.join(target_dir, json_name), 'w') as f:
                        json.dump(pixels, f)
                        
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    base_path: str = "data" 
    
    # Process Training Data
    print("--- Processing Training Data ---")
    process_directory(f"{base_path}/train/NORMAL", f"{base_path}/processed/train/normal")
    process_directory(f"{base_path}/train/PNEUMONIA", f"{base_path}/processed/train/pneumonia")
    
    # Process Testing Data
    print("\n--- Processing Testing Data ---")
    process_directory(f"{base_path}/test/NORMAL", f"{base_path}/processed/test/normal")
    process_directory(f"{base_path}/test/PNEUMONIA", f"{base_path}/processed/test/pneumonia")
    
    print("\nPre-processing complete!")