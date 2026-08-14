import os
from PIL import Image
from pathlib import Path

def crop_and_resize_images(directory, output_directory=None):
    """
    Read all image files from a directory and crop them to 1:1 aspect ratio,
    then resize to 200x200px.
    
    Args:
        directory: Path to the directory containing images
        output_directory: Path to save processed images (optional)
    """
    if output_directory is None:
        output_directory = os.path.join(directory, 'thumbnails')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    
    image_files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    for idx, filename in enumerate(image_files, 1):
        try:
            input_path = os.path.join(directory, filename)
            img = Image.open(input_path)
            
            
            width, height = img.size
            
            
            crop_size = min(width, height)
            
            
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            
            img_cropped = img.crop((left, top, right, bottom))
            
            
            img_resized = img_cropped.resize((200, 200), Image.Resampling.LANCZOS)
            
            
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_filename)
            img_resized.save(output_path, 'PNG')
            
            print(f"[{idx}/{len(image_files)}] Processed: {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nAll images saved to: {output_directory}")

if __name__ == "__main__":
    
    directory = input("Enter the directory path containing images: ")
    
    if os.path.isdir(directory):
        crop_and_resize_images(directory)
    else:
        print(f"Error: Directory '{directory}' not found")

