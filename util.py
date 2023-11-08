"""
Library containing utility functions to perform operations on data.
"""
import os
import cv2

RAW_DIR = os.path.join(os.curdir, 'raw')
PROCESSED_DIR = os.path.join(os.curdir, 'processed')


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
        else:
            print(f"{filename} is not in an image format")