import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 48
DATA_DIR = "dataset/train"

data = []
labels = []

# Get emotion labels (folders)
emotion_labels = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if not folder.startswith('.')
])

print("Emotion classes:", emotion_labels)

# -----------------------------
# Load & preprocess images
# -----------------------------
for label in emotion_labels:
    path = os.path.join(DATA_DIR, label)
    class_num = emotion_labels.index(label)

    for img in os.listdir(path):
        if img.startswith('.'):
            continue
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_array is None:
                continue

            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized)
            labels.append(class_num)

        except Exception as e:
            pass

# Convert to numpy arrays
X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(labels)

# Normalize pixel values
X = X / 255.0

print("Total samples:", X.shape[0])

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Training
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)
model.save("model/emotion_model.h5")

print("✅ Model training complete")
print("✅ Model saved at: model/emotion_model.h5")