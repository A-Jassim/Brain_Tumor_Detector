import os
import random
import shutil
import tensorflow as tf

def split_data(source_dir: str,
               dest_dir: str,
               categories: list,
               train_pct: float = 0.8,
               val_pct: float = 0.15,
               seed: int = 42):
    """Copie et répartit les images brutes en train/validation/test (80/15/5)."""
    random.seed(seed) #fixe le générateur de nombres aléatoires pour la reproductibilité

    for split in ['train', 'validation', 'test']:
        for cat in categories:
            os.makedirs(os.path.join(dest_dir, split, cat), exist_ok=True) #crée les répertoires de destination s'ils n'existent pas

    for cat in categories:
        imgs = os.listdir(os.path.join(source_dir, cat))
        random.shuffle(imgs) #on liste les images et on les mélange aléatoirement

        total = len(imgs)
        n_train = int(train_pct * total)
        n_val   = int(val_pct   * total)
        n_test  = total - n_train - n_val # assure que la somme des images est égale à total

        splits = {
            'train':      imgs[:n_train],
            'validation': imgs[n_train:n_train + n_val],
            'test':       imgs[n_train + n_val:]
        }

        for split, files in splits.items():
            """Pour chaque split, on copie les fichiers dans le répertoire approprié"""
            for fname in files:
                src = os.path.join(source_dir, cat, fname)
                dst = os.path.join(dest_dir, split, cat, fname)
                shutil.copy(src, dst)

    for split in ['train', 'validation', 'test']:
        for cat in categories:
            cnt = len(os.listdir(os.path.join(dest_dir, split, cat)))
            print(f"{split.capitalize()} — {cat}: {cnt} images") #on affiche le nombre d'images dans chaque catégorie et split

def load_datasets(data_dir: str,
                  image_size: tuple = (128, 128),
                  batch_size: int = 16,
                  seed: int = 42):
    """Charge les datasets TensorFlow depuis data_dir."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True, seed=seed
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True, seed=seed
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    return train_ds, val_ds, test_ds, train_ds.class_names

