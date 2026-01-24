import cv2
import os
import numpy as np

IMG_SIZE = 48
DATA_DIR = "dataset/train"

data = []
labels = []

emotion_labels = os.listdir(DATA_DIR)

for label in emotion_labels:
    path = os.path.join(DATA_DIR, label)
    class_num = emotion_labels.index(label)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized)
            labels.append(class_num)
        except Exception as e:
            pass

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)