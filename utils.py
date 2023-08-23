import numpy as np

def pad_images(images, target_shape):
    padded_images = []

    for image in images:
        old_shape = image.shape
        pad_width = [(0, max(0, new_dim - old_dim)) for old_dim, new_dim in zip(old_shape, target_shape)]
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        padded_images.append(padded_image)

    return np.array(padded_images)