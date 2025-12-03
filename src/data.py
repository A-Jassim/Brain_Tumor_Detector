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
    """Split images into train/validation/test (80/15/5)."""
    # Set random seed
    random.seed(seed)

    # Create directories
    # We need train, validation, and test folders for each category
    for split in ['train', 'validation', 'test']:
        for cat in categories:
            os.makedirs(os.path.join(dest_dir, split, cat), exist_ok=True)

    # Process each category
    for cat in categories:
        # Get and shuffle images
        imgs = os.listdir(os.path.join(source_dir, cat))
        random.shuffle(imgs)

        # Calculate split sizes
        total = len(imgs)
        n_train = int(train_pct * total)  # Example: 80% for training
        n_val   = int(val_pct   * total)  # Example: 15% for validation
        n_test  = total - n_train - n_val # Remaining for test (5%)

        # Divide into splits
        # Use list slicing to separate images
        splits = {
            'train':      imgs[:n_train],
            'validation': imgs[n_train:n_train + n_val],
            'test':       imgs[n_train + n_val:]
        }

        # Copy files to folders
        for split, files in splits.items():
            for fname in files:
                src = os.path.join(source_dir, cat, fname)
                dst = os.path.join(dest_dir, split, cat, fname)
                shutil.copy(src, dst)

    # Print counts
    for split in ['train', 'validation', 'test']:
        for cat in categories:
            cnt = len(os.listdir(os.path.join(dest_dir, split, cat)))
            print(f"{split.capitalize()} — {cat}: {cnt} images")

def load_datasets(data_dir: str,
                  image_size: tuple = (128, 128),
                  batch_size: int = 16,
                  seed: int = 42):
    """Load TensorFlow datasets."""
    # Load training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True, seed=seed
    )
    # Load validation data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True, seed=seed
    )
    # Load test data (no shuffle)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    # Return all datasets plus class names (e.g., ['yes', 'no'])
    return train_ds, val_ds, test_ds, train_ds.class_names

