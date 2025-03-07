from abc import ABC, abstractmethod

class BaseResizer(ABC):
    def __init__(self, target_char_size):
        self.target_char_size = target_char_size

    @abstractmethod
    def resize(self, image, bin_image):
        pass