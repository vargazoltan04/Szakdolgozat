import cv2
import numpy as np

from abc import ABC, abstractmethod
from .base_cleaner import BaseCleaner

class Cleaner(BaseCleaner):
    def delete_small_components(self, image, bin_image, min_size):
        image_inverted = cv2.bitwise_not(bin_image)
        num_labels, labels = cv2.connectedComponents(image_inverted)
        sizes = np.bincount(labels.ravel())
        for label in range(0, num_labels):
            if sizes[label] < min_size:
                labels[labels == label] = 0

        image = np.where(labels > 0, image, 255).astype(np.uint8)
        bin_image = np.where(labels > 0, bin_image, 255).astype(np.uint8)
        return image, bin_image