import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import sys

def load_and_preprocess(img_path: str, img_size: tuple):
    """Load and prepare image."""
    # Load and resize
    img = image.load_img(img_path, target_size=img_size)
    # Convert to array
    arr = image.img_to_array(img)
    # Add batch dimension
    # Model expects shape (batch_size, height, width, channels)
    return np.expand_dims(arr, axis=0)

def select_file():
    """Open file dialog to select image."""
    # Create window
    root = tk.Tk()
    # Hide window
    root.withdraw()
    # Select file
    file_path = filedialog.askopenfilename(
        title='Choose an image',
        filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
    )
    # Check if file selected
    if not file_path:
        print("No file selected.")
        sys.exit(1)
    return file_path

def main(args):
    """Load model and predict."""
    # Select image if not provided
    if args.img_path is None:
        args.img_path = select_file()

    # Load model
    model = load_model(args.model_path)
    # Prepare image
    x = load_and_preprocess(args.img_path, tuple(args.img_size))
    # Predict
    pred = float(model.predict(x)[0][0])
    # Get label
    # If prediction > 0.5, it's a tumor
    label = 'Tumor' if pred > 0.5 else 'No Tumor'
    # Show result
    print(f"Image: {args.img_path}")
    print(f"Prediction: {pred:.4f} → {label}")

if __name__ == '__main__':
    # Parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='models/best_model.keras')
    p.add_argument('--img_path',   type=str,)  # If None, will open file dialog
    p.add_argument('--img_size',   nargs=2, type=int, default=[128,128])
    args = p.parse_args()
    main(args)
