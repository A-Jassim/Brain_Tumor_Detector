import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

def build_model(input_shape=(128,128,3),
                dropout_rate=0.6,
                l2_rate=1e-4):
    """Build CNN model with MobileNetV2."""
    # Data augmentation
    # Random transformations to make model more robust
    data_aug = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ])

    # Load pre-trained MobileNetV2
    # Using weights trained on ImageNet for better performance
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    # Freeze base layers
    # We keep the pre-trained weights and only train our custom layers
    base_model.trainable = False

    # Build model
    model = tf.keras.Sequential([
        data_aug,                                      # Augmentation
        layers.Rescaling(1./255),                      # Normalize to [0, 1]
        base_model,                                    # Feature extractor
        layers.GlobalAveragePooling2D(),               # Pool features
        layers.Dense(64, activation='relu',            # Hidden layer
                     kernel_regularizer=regularizers.l2(l2_rate)),
        layers.Dropout(dropout_rate),                  # Prevent overfitting
        layers.Dense(1, activation='sigmoid')          # Binary output (0 or 1)
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def compute_class_weights(dataset):
    """Compute class weights for imbalanced data."""
    # Get all labels
    y = []
    for _, labels in dataset:  # Iterate through batches
        y.extend(labels.numpy())
    
    # Compute balanced weights
    # Give more weight to minority class to handle imbalance
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    # Return as dictionary {0: weight_0, 1: weight_1}
    return dict(enumerate(cw))


def get_callbacks(patience=12):
    """Return training callbacks."""
    return [
        EarlyStopping(
            monitor='val_accuracy',        # Watch validation accuracy
            mode='max',                    # Stop when it stops improving
            patience=patience,             # Wait this many epochs
            restore_best_weights=True      # Use best weights, not last
        )
    ]
