import cv2
import numpy as np

from abc import ABC, abstractmethod
from .base_cleaner import BaseCleaner

class Cleaner(BaseCleaner):
    def __init__(self, debug):
        self.debug = debug
        
    #Kis méretű elemek törlése
    #ezt úgy teszi meg, hogy megkeresi a min_size-nál kisebb elemeket, 
    #és azokat kitörli
    def delete_small_components(self, image, min_size):
        image_inverted = cv2.bitwise_not(image)
        num_labels, labels = cv2.connectedComponents(image_inverted)

        sizes = np.bincount(labels.ravel())
        for label in range(0, num_labels):
            if sizes[label] < min_size:
                labels[labels == label] = 0

        image = np.where(labels > 0, image, 255).astype(np.uint8)
        return image