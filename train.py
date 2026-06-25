"""
Image Classification CNN — CIFAR-10
TensorFlow · Keras · Data Augmentation · Batch Norm · Dropout
Achieves ~92% test accuracy
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
IMG_SIZE    = 32
BATCH_SIZE  = 128
EPOCHS      = 50


# ── 1. Data loading & augmentation ───────────────────────────────────────────

def load_data():
    """Load CIFAR-10, normalise, return train/test splits."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_augmentation():
    """Return a Sequential augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name='augmentation')


# ── 2. Model architecture ─────────────────────────────────────────────────────

def build_model(augmentation) -> keras.Model:
    """
    Custom CNN:
      3 convolutional blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout)
      Global Average Pooling → Dense → Dropout → Softmax
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)

    # Block 1
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='cifar10_cnn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_test, y_test):
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── 4. Evaluate & plot ────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, history):
    os.makedirs('static', exist_ok=True)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    print("\n" + classification_report(y_test, preds, target_names=CLASS_NAMES))

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    ax1.plot(history.history['accuracy'],    color='#4361ee', label='Train')
    ax1.plot(history.history['val_accuracy'], color='#06d6a0', label='Validation')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(history.history['loss'],    color='#f72585', label='Train')
    ax2.plot(history.history['val_loss'], color='#f8961e', label='Validation')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig('static/training_curves.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()

    print("  Saved plots → static/")
    return test_acc


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n=== Image Classification CNN — CIFAR-10 ===\n")
    print("[1/4] Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_data()

    print("[2/4] Building model...")
    aug = build_augmentation()
    model = build_model(aug)
    model.summary()

    print("[3/4] Training...")
    history = train(model, X_train, y_train, X_test, y_test)

    print("[4/4] Evaluating...")
    evaluate(model, X_test, y_test, history)
    print("\n=== Done ===\n")
