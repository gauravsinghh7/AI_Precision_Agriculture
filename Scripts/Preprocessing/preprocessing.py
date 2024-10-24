#Steps for Preprocessing in Python
#Image Loading
#Image Cleaning
#Image Resizing
#Image Normalization
#Data Augmentation (Optional but recommended) 


#Image Loading

import os
import cv2  # OpenCV for image processing
import numpy as np

# Load images from the folder and its subfolders
def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):  # Walk through all subdirectories
        for filename in files:
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Path to the folder with raw images
raw_images_path = r"C:\Users\Offic\OneDrive\Desktop\AI_Precision_Agriculture\Data\Raw_Images"

raw_images = load_images_from_folder(raw_images_path)
print(f"Loaded {len(raw_images)} images from {raw_images_path}")


# Apply Gaussian Blur to remove noise
def clean_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Clean all raw images
cleaned_images = [clean_image(img) for img in raw_images]


cleaned_images_path = "AI_Precision_Agriculture/Data/Cleaned_Images/"

# Save cleaned images to folder
def save_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for idx, img in enumerate(images):
        filename = f"cleaned_image_{idx}.jpg"
        cv2.imwrite(os.path.join(folder, filename), img)

save_images(cleaned_images, cleaned_images_path)


# Resize images to a standard size (e.g., 256x256 pixels)
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Resize all cleaned images
resized_images = [resize_image(img) for img in cleaned_images]

# Save resized images
resized_images_path = "AI_Precision_Agriculture/Data/Resized_Images/"
save_images(resized_images, resized_images_path)

#Image Normalization


# Normalize image pixel values to the range [0, 1]
def normalize_image(image):
    return image / 255.0

# Normalize all resized images
normalized_images = [normalize_image(img) for img in resized_images]

# Save normalized images (optional, or just use them in memory)
normalized_images_path = "AI_Precision_Agriculture/Data/Normalized_Images/"
save_images([np.uint8(img * 255) for img in normalized_images], normalized_images_path)  # Saving as uint8 format


# Data Augmentation 

from keras.preprocessing.image import ImageDataGenerator

# Create an image data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply augmentation to a sample image
sample_image = resized_images[0].reshape((1,) + resized_images[0].shape)  # Keras expects 4D input
i = 0
for batch in datagen.flow(sample_image, batch_size=1):
    aug_img = batch[0].astype(np.uint8)
    cv2.imwrite(f"AI_Precision_Agriculture/Data/Augmented_Images/aug_image_{i}.jpg", aug_img)
    i += 1
    if i > 10:  # Save 10 augmented images
        break
