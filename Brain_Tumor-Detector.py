import cv2
import os
import pandas as pd
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


#permet de savoir et de connaitre comment nos images et nos données sont organisées

dict={}
fichiers_yes= os.listdir('data_set/yes')
fichiers_no= os.listdir('data_set/no')
img   = plt.imread('data_set/yes/Y6.jpg')

for fichier in fichiers_yes:
    dict[fichier] = (img.shape[0], img.shape[1], img.shape[2])
for fichier in fichiers_no:
    dict[fichier] = (img.shape[0], img.shape[1], img.shape[2])

data =pd.DataFrame(dict)
print(data.head())
print(data.describe())
img_r=cv2.resize(img, (75, 75))
plt.imshow(img_r)
plt.axis('off')
plt.show()

# on va maintenant ranger nos images dans différent dossiers pour avoir un data set de validations et un data set d'entrainement
os.makedirs('data_set/train', exist_ok=True)
os.makedirs('data_set/train/yes', exist_ok=True)
os.makedirs('data_set/train/no', exist_ok=True)
os.makedirs('data_set/validation', exist_ok=True)
os.makedirs('data_set/validation/yes', exist_ok=True)
os.makedirs('data_set/validation/no', exist_ok=True)

source_yes = 'data_set/yes'   
source_no = 'data_set/no'

# Dossiers de destination
dest_base = 'data_set'
categories = ['yes', 'no']

# Pour chaque catégorie (yes, no)
for cat in categories:
    source_dir = os.path.join('data_set', cat)
    images = os.listdir(source_dir)
    random.shuffle(images)  # mélange aléatoire

    total = len(images)
    split_index = int(0.8 * total)

    # Séparation
    train_files = images[:split_index]
    val_files = images[split_index:]

    # Copier vers train
    for img in train_files:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(dest_base, 'train', cat, img)
        shutil.copy(src_path, dst_path)

    # Copier vers validation
    for img in val_files:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(dest_base, 'validation', cat, img)
        shutil.copy(src_path, dst_path)

nb_fichier_yes = len(os.listdir('data_set/train/yes'))
nb_fichier_no = len(os.listdir('data_set/train/no'))
print(f"Nombre de fichiers 'yes' dans le dossier d'entraînement : {nb_fichier_yes}")
print(f"Nombre de fichiers 'no' dans le dossier d'entraînement : {nb_fichier_no}")
nb_fichier_yes_val = len(os.listdir('data_set/validation/yes'))
nb_fichier_no_val = len(os.listdir('data_set/validation/no'))
print(f"Nombre de fichiers 'yes' dans le dossier de validation : {nb_fichier_yes_val}")
print(f"Nombre de fichiers 'no' dans le dossier de validation : {nb_fichier_no_val}")

train_data=tf.keras.preprocessing.image_dataset_from_directory(
    'data_set/train',
    image_size=(75, 75),
    batch_size=16,
    shuffle=True,
    seed=42
)

val_data=train_data=tf.keras.preprocessing.image_dataset_from_directory(
    'data_set/validation',
    image_size=(75, 75),
    batch_size=16,
    shuffle=False,
)

#on regarde les classes de nos données
class_names = val_data.class_names
print(class_names)

#on regarde a quoi ressemble un batch
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(16):
    ax = plt.subplot(1, 16, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

#on construit notre modèle

model = tf.keras.Sequential([
   layers.Rescaling(1./255),
   layers.Conv2D(128,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(64,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(32,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(16,4, activation='relu'),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(64, activation='relu'),
   layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'],)
    
history=model.fit( 
  train_data,
  validation_data=val_data,
  epochs=30
)



model.summary(expand_nested=True)
# Accuracy
plt.plot(history.history['accuracy'], label='Entraînement')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Entraînement')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

y_true = []
y_pred = []

for images, labels in val_data:
    predictions = model.predict(images)
    # Seuil à 0.5 pour classifier
    preds = (predictions > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

# Affichage du rapport de classification
print(classification_report(y_true, y_pred, target_names=class_names))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()