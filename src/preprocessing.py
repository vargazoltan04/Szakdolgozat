import cv2
import numpy as np

def binarize(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)

    return image_binary

def delete_small_components(image_bin, min_size):
    image_inverted = cv2.bitwise_not(image_bin)
    num_labels, labels = cv2.connectedComponents(image_inverted)
    sizes = np.bincount(labels.ravel())
    for label in range(0, num_labels):
        if sizes[label] < min_size:
            labels[labels == label] = 0

    binary_output = np.where(labels > 0, 255, 0).astype(np.uint8)
    binary_output = cv2.bitwise_not(binary_output)
    return binary_output