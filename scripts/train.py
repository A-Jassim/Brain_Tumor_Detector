import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
from src.data import split_data, load_datasets
from src.model import build_model, compute_class_weights, get_callbacks
from src.evaluate import plot_metrics, evaluate_model

def main(args):
    # Initialisation aléatoire
    random.seed(42)

    # Analyse des dimensions des images brutes
    image_dims = {}
    for cat in args.categories:
        src_cat = os.path.join(args.raw_dir, cat)
        for fname in os.listdir(src_cat):
            img = plt.imread(os.path.join(src_cat, fname))
            image_dims[f"{cat}/{fname}"] = img.shape
    df_dims = pd.DataFrame.from_dict(
        image_dims, orient='index', columns=['height','width','channels']
    )
    print(df_dims.head())
    print(f"Total images: {df_dims.shape[0]}")

    # Affichage d'exemple redimensionné
    sample_img = plt.imread(
        os.path.join(args.raw_dir, args.categories[0],
                     os.listdir(os.path.join(args.raw_dir, args.categories[0]))[0])
    )
    img_r = cv2.resize(sample_img, tuple(args.img_size))
    plt.imshow(img_r)
    plt.axis('off')
    plt.show()

    # Préparation des données
    if not os.path.isdir(os.path.join(args.processed_dir, 'train')):
        split_data(
            source_dir=args.raw_dir,
            dest_dir=args.processed_dir,
            categories=args.categories
        )
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.processed_dir,
        image_size=tuple(args.img_size),
        batch_size=args.batch_size
    )

    # Construction et entraînement du modèle
    model = build_model(
        input_shape=(*args.img_size, 3),
        dropout_rate=args.dropout,
        l2_rate=args.l2
    )
    cw = compute_class_weights(train_ds)
    callbacks = get_callbacks(patience=args.patience)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=cw
    )

    # Affichage des métriques et rapports
    plot_metrics(history)
    evaluate_model(model, train_ds, class_names, title='Train')
    evaluate_model(model, val_ds, class_names, title='Validation')
    evaluate_model(model, test_ds, class_names, title='Test')

    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir',        type=str, default='raw_data')
    p.add_argument('--processed_dir',  type=str, default='data_set')
    p.add_argument('--categories',     nargs='+', required=True,
                   help='ex: yes no')
    p.add_argument('--img_size',       nargs=2, type=int, default=[128,128])
    p.add_argument('--batch_size',     type=int, default=16)
    p.add_argument('--epochs',         type=int, default=20)
    p.add_argument('--dropout',        type=float, default=0.6)
    p.add_argument('--l2',             type=float, default=1e-4)
    p.add_argument('--patience',       type=int, default=12)
    p.add_argument('--output_model',   type=str, default='models/best_model.keras')
    args = p.parse_args()
    main(args)