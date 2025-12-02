# Street View Geolocation Predictor

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to predict **latitude and longitude** from **street-level panoramic images**. The images are preprocessed and fed into the CNN for a regression task.

---

## 📂 Files Included

### `train.py`
- Main training script.
- Loads, preprocesses, and splits the data.
- Defines the CNN architecture and compiles the model.
- Loads pre-trained weights.
- Trains the model and evaluates performance.

### `preprocess.py`
- Preprocesses raw images from the `imgs/` directory.
- Combines four separate street-view images into one long panoramic image.
- Outputs results to `preprocessed_imgs/`.

### `evalute.py`
- Evaluates the trained model on a test set.
- Can also run a prediction on a single example.

### `collectImgs.py`
- Uses Selenium and PyAutoGUI to automate image collection from an online map service.
- Saves both images and associated latitude/longitude data.

### Additional Files/Directories
- **`latLong.csv`** – Contains latitude/longitude pairs for each preprocessed image.
- **`preprocessed_imgs/`** – Contains `200 × 2000` grayscale panoramic images.
- **`imgs/`** – Raw captured street-view image segments.
- **`checkpoints/`** – Model weight files (`.ckpt`) saved during training.

---

## ⚙️ CNN Architecture

The model is a deep CNN designed to extract visual features from wide panoramic input images and output a pair: [longitude, latitude]


### **Input Shape**
(200, 2000, 1) # Height × Width × Channels (grayscale)

### **Layer Overview**

| Layer Type | Filters/Units | Kernel Size | Stride | Activation | Output Shape (Approx.) | Notes |
|-----------|----------------|-------------|--------|------------|-------------------------|-------|
| **Input** | – | – | – | – | (200, 2000, 1) | – |
| Conv2D | 32 | (3, 3) | – | ReLU | (200, 2000, 32) | padding="same" |
| MaxPool2D | – | (2, 2) | (2, 2) | – | (100, 1000, 32) | – |
| Dropout | – | – | – | – | (100, 1000, 32) | rate=0.5 |
| Conv2D | 64 | (3, 3) | – | ReLU | (100, 1000, 64) | padding="same" |
| MaxPool2D | – | (2, 2) | (2, 2) | – | (50, 500, 64) | – |
| Dropout | – | – | – | – | (50, 500, 64) | rate=0.5 |
| Conv2D | 64 | (3, 3) | – | ReLU | (50, 500, 64) | padding="same" |
| MaxPool2D | – | (2, 2) | (2, 2) | – | (25, 250, 64) | – |
| Dropout | – | – | – | – | (25, 250, 64) | rate=0.5 |
| Conv2D | 64 | (3, 3) | – | ReLU | (25, 250, 64) | padding="same" |
| MaxPool2D | – | (5, 5) | (5, 5) | – | (5, 50, 64) | – |
| Dropout | – | – | – | – | (5, 50, 64) | rate=0.5 |
| Conv2D | 128 | (3, 3) | – | ReLU | (5, 50, 128) | padding="same" |
| Flatten | – | – | – | – | (32000,) | – |
| Dense | 100 | – | – | ReLU | (100,) | – |
| Output Dense | 2 | – | – | Linear | (2,) | `[lat, lon]` |

### **Training Configuration**
- **Loss:** `mean_squared_error`
- **Optimizer:** `adam`

---

## ▶️ Getting Started

### **1. Data Collection (Optional)**
Run the automated scraper:

```bash
python collectImgs.py
```

2. Preprocessing
```bash
python preprocess.py
```
Creates combined panoramic images in preprocessed_imgs/.

3. Training
```bash
python train.py
```

Loads/continues from checkpoint cp-00030.ckpt.

Trains for 100 epochs.

Saves new checkpoints every 5 steps.

4. Evaluation
```bash
python evalute.py
```
Tests performance and runs sample predictions.

