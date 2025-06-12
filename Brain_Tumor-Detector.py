import cv2
import os
import pandas as pd
import random
import shutil
import numpy as np
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# Pour que le shuffle soit toujours identique à chaque exécution :
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------------------------
# 1) Informations sur les dimensions d'image
# -------------------------------------------------------------------
image_dims = {}
fichiers_yes = os.listdir('data_set/yes')
fichiers_no  = os.listdir('data_set/no')
img = plt.imread('data_set/yes/Y6.jpg')

for fichier in fichiers_yes + fichiers_no:
    image_dims[fichier] = (img.shape[0], img.shape[1], img.shape[2])

data = pd.DataFrame(image_dims)
print(data.head())
print(data.shape)

img_r = cv2.resize(img, (128, 128))
plt.imshow(img_r)
plt.axis('off')
plt.show()

# -------------------------------------------------------------------
# 2) Création des répertoires train/validation/test
# -------------------------------------------------------------------
base_dirs  = ['train', 'validation', 'test']
categories = ['yes', 'no']
for base in base_dirs:
    for cat in categories:
        os.makedirs(f'data_set/{base}/{cat}', exist_ok=True)

for cat in categories:
    source_dir = os.path.join('data_set', cat)
    images = os.listdir(source_dir)
    random.shuffle(images)
    total   = len(images)
    n_train = int(0.8 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val

    train_files = images[:n_train]
    val_files   = images[n_train:n_train + n_val]
    test_files  = images[n_train + n_val:]

    print(f"\n{cat.upper()} — Total: {total} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    for img_name in train_files:
        shutil.copy(os.path.join(source_dir, img_name), f'data_set/train/{cat}/{img_name}')
    for img_name in val_files:
        shutil.copy(os.path.join(source_dir, img_name), f'data_set/validation/{cat}/{img_name}')
    for img_name in test_files:
        shutil.copy(os.path.join(source_dir, img_name), f'data_set/test/{cat}/{img_name}')

# Vérification des tailles
for base in base_dirs:
    for cat in categories:
        count = len(os.listdir(f'data_set/{base}/{cat}'))
        print(f"{base.upper()} - {cat}: {count} images")

# -------------------------------------------------------------------
# 3) Chargement des datasets statiques (pour la visualisation & le test final)
# -------------------------------------------------------------------
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data_set/train', image_size=(128, 128), batch_size=16, shuffle=True, seed=42
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data_set/validation', image_size=(128, 128), batch_size=16, shuffle=False
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data_set/test', image_size=(128, 128), batch_size=16, shuffle=False
)

class_names = test_data.class_names
print("Classes :", class_names)

# Visualisation rapide
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(16):
        ax = plt.subplot(1, 16, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# -------------------------------------------------------------------
# 4) Définition du modèle
# -------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False

def build_model():
    model = tf.keras.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=12)

# -------------------------------------------------------------------
# 5) Calcul des poids de classes
# -------------------------------------------------------------------
y_train = []
for _, labels in train_data:
    y_train.extend(labels.numpy())
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Poids des classes :", class_weight_dict)

# -------------------------------------------------------------------
# 6) *** Validation croisée Stratified K-Fold sur `data_set/train` ***
# -------------------------------------------------------------------
# 6a) Rassembler chemins et labels du dossier `data_set/train`
train_paths  = []
train_labels = []
for cat, lab in [('yes', 1), ('no', 0)]:
    dir_train_cat = os.path.join('data_set', 'train', cat)
    for fname in os.listdir(dir_train_cat):
        train_paths.append(os.path.join(dir_train_cat, fname))
        train_labels.append(lab)
train_paths  = np.array(train_paths)
train_labels = np.array(train_labels)

# 6b) Initialisation du split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

# 6c) Boucle sur chaque fold
for train_idx, val_idx in skf.split(train_paths, train_labels):
    print(f"\n=== K-Fold CV: Fold {fold_no} ===")

    # Construire les tf.data.Dataset pour ce fold
    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [128, 128])
        return img, label

    fold_train_ds = (
        tf.data.Dataset
          .from_tensor_slices((train_paths[train_idx], train_labels[train_idx]))
          .shuffle(len(train_idx), seed=42)
          .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(16).prefetch(tf.data.AUTOTUNE)
    )
    fold_val_ds = (
        tf.data.Dataset
          .from_tensor_slices((train_paths[val_idx], train_labels[val_idx]))
          .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(16).prefetch(tf.data.AUTOTUNE)
    )

    # (Re)construction et entraînement du modèle
    model_k = build_model()
    model_k.fit(
        fold_train_ds,
        validation_data=fold_val_ds,
        epochs=20,
        callbacks=[es],
        class_weight=class_weight_dict
    )

    # Évaluation du fold
    y_true_f, y_pred_f = [], []
    for imgs, labs in fold_val_ds:
        preds = (model_k.predict(imgs) > 0.5).astype(int).flatten()
        y_pred_f.extend(preds)
        y_true_f.extend(labs.numpy())

    # Rapport détaillé
    print("Rapport validation fold :")
    print(classification_report(y_true_f, y_pred_f, target_names=class_names))

    # Précision du fold
    acc_val = accuracy_score(y_true_f, y_pred_f)
    print(f"Précision (accuracy) fold {fold_no} : {acc_val:.2f}")

    # Matrice de confusion du fold
    cm_val = confusion_matrix(y_true_f, y_pred_f)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title(f'Matrice de confusion (Fold {fold_no})')
    plt.show()

    fold_no += 1


# -------------------------------------------------------------------
# 7) Évaluation finale sur le jeu `data_set/test` (inchangé)
# -------------------------------------------------------------------
print("\n=== Évaluation finale sur test_data ===")
y_true_test, y_pred_test = [], []
for imgs, labs in test_data:
    preds = (model_k.predict(imgs) > 0.5).astype(int).flatten()
    y_pred_test.extend(preds)
    y_true_test.extend(labs.numpy())

print(classification_report(y_true_test, y_pred_test, target_names=class_names))

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
y_true_val_final, y_pred_val_final = [], []
for imgs, labs in val_data:
    preds = (model_k.predict(imgs) > 0.5).astype(int).flatten()
    y_pred_val_final.extend(preds)
    y_true_val_final.extend(labs.numpy())

acc_val_final = accuracy_score(y_true_val_final, y_pred_val_final)
print(f'Précision finale sur validation set : {acc_val_final:.2f}')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion (Test set)')
plt.show()

from sklearn.metrics import accuracy_score

acc_test_final = accuracy_score(y_true_test, y_pred_test)
print(f'Précision finale sur test set : {acc_test_final:.2f}')