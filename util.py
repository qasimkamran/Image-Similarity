"""
Library containing utility functions to perform operations on data.
"""
import os
import cv2
import numpy as np
from keras.applications import resnet_v2

RAW_DIR = os.path.join(os.curdir, 'raw')
PROCESSED_DIR = os.path.join(os.curdir, 'processed')
PREPROCESSED_DIR = os.path.join(os.curdir, 'preprocessed')


def homogenize_raw_images(width, height, image_format):
    for filename in os.listdir(RAW_DIR):
        filepath = os.path.join(RAW_DIR, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image = cv2.imread(filepath)
            resized_image = cv2.resize(image, (width, height))
            file_extension = os.path.splitext(filename)[-1].lower()
            output_filename = os.path.splitext(filename)[0] + f'.{image_format}'
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            cv2.imwrite(output_path, resized_image)
            print(f"Written {output_path}")
        else:
            print(f"{filename} is not in an image format")


def preprocess_resnet50():
    target_size = (224, 224)
    for filename in os.listdir(PROCESSED_DIR):
        filepath = os.path.join(PROCESSED_DIR, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image = cv2.imread(filepath)
            resized_image = cv2.resize(image, target_size)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            preprocessed_image = resnet_v2.preprocess_input(np.expand_dims(rgb_image, axis=0))
            filename = os.path.splitext(filename)[0]
            output_path = os.path.join(PREPROCESSED_DIR, filename)
            np.save(output_path, preprocessed_image)
            print(f"Written {output_path}.npy")
        else:
            print(f"{filename} is not in an image format")
