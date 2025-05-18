from abc import ABC, abstractmethod

class BaseRowSegmentator(ABC):
    def __init__(self, debug):
        self.debug = debug
        
    @abstractmethod
    def row_segmentation(self, image, bin_image):
        pass