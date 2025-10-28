import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import sys

def load_and_preprocess(img_path: str, img_size: tuple):
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img)
    return np.expand_dims(arr, axis=0)

def select_file():
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale
    file_path = filedialog.askopenfilename(
        title='Choisir une image',
        filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
    )
    if not file_path:
        print("Aucun fichier sélectionné.")
        sys.exit(1)
    return file_path

def main(args):
    if args.img_path is None:
        args.img_path = select_file()

    model = load_model(args.model_path)
    x = load_and_preprocess(args.img_path, tuple(args.img_size))
    pred = float(model.predict(x)[0][0])
    label = 'Tumor' if pred > 0.5 else 'No Tumor'
    print(f"Image: {args.img_path}")
    print(f"Prediction: {pred:.4f} → {label}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='models/best_model.keras')
    p.add_argument('--img_path',   type=str,)
    p.add_argument('--img_size',   nargs=2, type=int, default=[128,128])
    args = p.parse_args()
    main(args)
