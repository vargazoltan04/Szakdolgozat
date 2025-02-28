from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    @abstractmethod
    def visualize_confusion_matrix(self, true, prediction):
        pass