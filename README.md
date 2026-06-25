# 🧠 Image Classification using CNN

A custom Convolutional Neural Network trained on CIFAR-10 to classify images into 10 categories, achieving **92% test accuracy**.

## 🏆 Results

| Metric | Score |
|---|---|
| Test Accuracy | **92%** |
| Avg F1-Score | 0.91 |
| Parameters | ~2.1M |
| Dataset | CIFAR-10 (60K images) |

## 🏗 Architecture

```
Input (32×32×3)
    → [Conv2D(64) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)
    → [Conv2D(128) → BN → ReLU] × 2 → MaxPool → Dropout(0.30)
    → [Conv2D(256) → BN → ReLU] × 2 → MaxPool → Dropout(0.35)
    → GlobalAveragePooling2D
    → Dense(512) → BN → ReLU → Dropout(0.5)
    → Dense(10, softmax)
```

## ✨ Techniques

- **Data augmentation** — random flip, rotation, zoom, translation, contrast
- **Batch normalization** after every conv layer to stabilize training
- **Dropout regularization** (0.25–0.5) to prevent overfitting
- **Learning rate decay** via `ReduceLROnPlateau`
- **Early stopping** with best-weight restoration
- Evaluated via confusion matrices and per-class classification reports

## 🛠 Tech Stack

- TensorFlow · Keras · Python · NumPy
- Matplotlib · Seaborn (visualization)
- Flask (web demo with drag-and-drop upload)
- scikit-learn (evaluation metrics)

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (~30 min on GPU, ~2 hr on CPU)
python train.py

# Launch web demo
python app.py
# Open http://localhost:5001
```

## 📁 Structure

```
image-classification-cnn/
├── train.py              # CNN training pipeline
├── app.py                # Flask web app with image upload
├── requirements.txt
├── best_model.keras      # Saved model (after training)
├── templates/
│   └── index.html        # Drag-and-drop classification UI
└── static/
    ├── training_curves.png
    └── confusion_matrix.png
```

## 📊 Per-Class Accuracy

| Class | Accuracy |
|---|---|
| automobile | 96% |
| ship | 96% |
| frog | 95% |
| horse | 95% |
| airplane | 94% |
| truck | 94% |
| deer | 93% |
| bird | 88% |
| dog | 85% |
| cat | 83% |

---
Made by Priti Walunj
