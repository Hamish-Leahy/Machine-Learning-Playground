from PIL import Image
import numpy as np

# Load your input image
input_image = Image.open('input_image.jpg')

# Define custom data augmentation functions
def rotate_image(image, angle_degrees):
    return image.rotate(angle_degrees)

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# Apply data augmentation
augmented_images = []

# Rotate the image by 90, 180, and 270 degrees
for angle in [90, 180, 270]:
    rotated_image = rotate_image(input_image, angle)
    augmented_images.append(rotated_image)

# Flip the image horizontally
flipped_image = flip_image(input_image)
augmented_images.append(flipped_image)

# Save augmented images
for i, augmented_image in enumerate(augmented_images):
    augmented_image.save(f'augmented_image_{i}.jpg')
