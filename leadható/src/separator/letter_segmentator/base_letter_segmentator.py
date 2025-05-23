from abc import ABC, abstractmethod

class BaseLetterSegmentator(ABC):
    def __init__(self, debug):
        self.debug = debug

    @abstractmethod
    def letter_segmentation(self, row, bin_row):
        pass