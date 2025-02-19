from abc import ABC, abstractmethod

class BaseRowSegmentator(ABC):
    @abstractmethod
    def row_segmentation(self, image, bin_image):
        pass