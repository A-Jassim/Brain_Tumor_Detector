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

# Pour que le shuffle soit toujours identique à chaque exécution :
random.seed(42)

# Informations sur les dimensions d'image
image_dims = {}
fichiers_yes = os.listdir('data_set/yes')
fichiers_no = os.listdir('data_set/no')
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

# Création des répertoires (exist_ok=True évite l'erreur si le dossier existe déjà)
base_dirs = ['train', 'validation', 'test']
categories = ['yes', 'no']
for base in base_dirs:
    for cat in categories:
        os.makedirs(f'data_set/{base}/{cat}', exist_ok=True)

# Répartition des fichiers (80% train, 10% val, 10% test), avec shuffle reproductible
for cat in categories:
    source_dir = os.path.join('data_set', cat)
    images = os.listdir(source_dir)
    random.shuffle(images)  # shuffle() utilisera la graine fixée ci-dessus

    total = len(images)
    n_train = int(0.8 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val  # le reste va dans test

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

# Chargement des datasets
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

# Visualisation des images
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(16):
        ax = plt.subplot(1, 16, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Data augmentation
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
base_model.trainable = False  # Freeze for initial training

model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])

es = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=12
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Extraire tous les labels du train_data (TensorFlow Dataset)
y_train = []
for _, labels in train_data:
    y_train.extend(labels.numpy())

# Calculer les poids inverses des classes (équilibrage)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights_array))
print("Poids des classes :", class_weight_dict)

# Entraînement avec gestion des poids de classes
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[es],
    class_weight=class_weight_dict
)

# Courbes
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.show()
y_true_train, y_pred_train = [], []
for imgs, labs in train_data:
    preds = model.predict(imgs)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred_train.extend(preds)
    y_true_train.extend(labs.numpy())
print("Train set report:")
print(classification_report(y_true_train, y_pred_train, target_names=class_names))

# 2) Évaluer sur val_data
y_true_val, y_pred_val = [], []
for imgs, labs in val_data:
    preds = model.predict(imgs)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred_val.extend(preds)
    y_true_val.extend(labs.numpy())
print("Validation set report:")
print(classification_report(y_true_val, y_pred_val, target_names=class_names))

# Évaluation sur le test set
y_true, y_pred = [], []
for images, labels in test_data:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

print("\nRapport de classification sur le test set :\n")
print(classification_report(y_true, y_pred, target_names=class_names))
accuracy = accuracy_score(y_true_val, y_pred_val)
print('Val Accuracy = %.2f' % accuracy)
# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion (Test set)')
plt.show()
