from abc import ABC, abstractmethod

class BaseResizer(ABC):
    @abstractmethod
    def resize(self, image, bin_image):
        pass