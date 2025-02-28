from abc import ABC, abstractmethod
from separator.visualizer.base_visualizer import BaseVisualizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np

class Visualizer(BaseVisualizer):
     def __init__(self, save_path):
          self.save_path = save_path
     
     def visualize_confusion_matrix(self, true, prediction, file_name, normalize=True):
          min_length = min(len(true), len(prediction))
          true = true[:min_length]
          prediction = prediction[:min_length]

          allowed_chars = string.ascii_letters
          true = [c for c in true if c in allowed_chars]
          prediction = [c for c in prediction if c in allowed_chars]

          unique_labels = sorted(set(true) | set(prediction))
          cm = confusion_matrix(true, prediction, labels=unique_labels)

          if normalize:
               cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize row-wise
               cm = np.nan_to_num(cm)  # Replace NaN values (from division by zero) with 0

          print(cm)

          # Ábra méretének beállítása
          plt.figure(figsize=(6,5))

          fmt = '.2f' if normalize else 'd'
          # Seaborn heatmap használata a mátrix vizualizálásához
          sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)

          # Rotate labels for readability
          plt.xticks(rotation=90)
          plt.yticks(rotation=0)

          # Címek beállítása
          plt.xlabel("Predicted labels")
          plt.ylabel("True labels")
          plt.title("Confusion Matrix")

          # Kép mentése fájlba
          plt.savefig(f"{self.save_path}/{file_name}", dpi=300, bbox_inches='tight')

          # Opcionálisan megjelenítheted is
          plt.show()

