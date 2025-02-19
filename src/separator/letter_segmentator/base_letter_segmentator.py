from abc import ABC, abstractmethod

class BaseLetterSegmentator(ABC):
    @abstractmethod
    def letter_segmentation(self, row, bin_row):
        pass