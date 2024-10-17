import os
import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov11.pt")

def get_image_paths(folder_path):
    # Define image file extensions (you can add more if needed)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

    # Collect all image file paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_paths

def imshow_grid(image_batches, col=10, save_path=None):
    batch_size = len(image_batches)
    col = min(col, batch_size)

    row = math.ceil(batch_size / col)
    
    mult = 20
    fig, axs = plt.subplots(row, col, figsize=(col * mult, row * mult))
    fig.patch.set_facecolor('black')
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        if i < batch_size:
            image = image_batches[i]
            # print(type(image))
            image = np.asarray(image)
            # image = image.permute(1, 2, 0).detach().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            ax.imshow(image)
        ax.axis("off")
    if save_path is not None:
        plt.savefig(save_path + "results.png")
    else:
        plt.show()

def predict_from_path(image_path, model):
    model = YOLO(model)
    results = model.predict(image_path)
    predictions = results[0].boxes
    class_names = results[0].names

    image = np.asarray(Image.open(image_path))
    image = np.copy(image)

    for box in predictions:
        class_idx = int(box.cls[0])
        class_name = class_names[class_idx]
        confidence = box.conf[0].item()
        confidence = round(confidence, 2)
        # print(confidence)
        bbox = box.xyxy[0].tolist()
        image.setflags(write=1)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.putText(image, str(confidence), (x1, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, class_name, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        image = np.asarray(image)
    
    return image

def prediction():
    file_num = 5
    folder_path = 'images/upload/'
    image_files = get_image_paths(folder_path)
    random.shuffle(image_files)
    image_files = image_files[:file_num]
    
    image_results = []

    for image_path in image_files:
        # img_path = "datasets/train/images/2C0.jpg"
        results = model.predict(image_path)
        predictions = results[0].boxes
        class_names = results[0].names

        image = np.asarray(Image.open(image_path))
        image = np.copy(image)

        for box in predictions:
            class_idx = int(box.cls[0])
            class_name = class_names[class_idx]
            confidence = box.conf[0] 
            bbox = box.xyxy[0].tolist()    
            image.setflags(write=1)
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            image = np.asarray(image)
            image_results.append(image)
            plt.imshow(image)
        plt.show()

    imshow_grid(image_results, save_path="results/")

def predict():
    file_num = 5
    folder_path = 'datasets/test'
    image_files = get_image_paths(folder_path)
    random.shuffle(image_files)
    image_files = image_files[:file_num]

    image_results = []

    for image_path in image_files:
        # img_path = "datasets/train/images/2C0.jpg"
        image = predict_from_path(image_path)
        plt.imshow(image)
    plt.show()
    
    imshow_grid(image_results, save_path="results/")

# image = predict_from_path("images/upload/uqSCLaApLMFzYZmmAXAPomygPgLbpceaXUZJtkZchfWqcbLyVcASlvTvPyoKvwRY.jpg")
# plt.imshow(image)
# plt.show()
# plt.imsave("images/result/a.jpg", image)

# prediction()