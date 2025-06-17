import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

def build_model(input_shape=(128,128,3),
                dropout_rate=0.6,
                l2_rate=1e-4):
    """Construit et compile un modèle CNN avec MobileNetV2 en base."""
    data_aug = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        data_aug,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(l2_rate)),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def compute_class_weights(dataset):
    """Calcule les poids inverses par classe pour gérer le déséquilibre."""
    y = []
    for _, labels in dataset:
        y.extend(labels.numpy())
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(cw))


def get_callbacks(patience=12):
    """Retourne la liste de callbacks (EarlyStopping)."""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=patience,
            restore_best_weights=True
        )
    ]
