from abc import ABC, abstractmethod
from separator.visualizer.base_visualizer import BaseVisualizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np

import Levenshtein
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from util import util

class Visualizer(BaseVisualizer):
    def __init__(self, save_path):
         self.save_path = save_path
   
    def visualize_confusion_matrix(self, labels, true, prediction, normalize = False):
        # 2. Konfúziós mátrix készítése
        conf_matrix, labels = self.generate_confusion_matrix(labels, true, prediction, normalize)

        # 3. Megjelenítés
        self.plot_confusion_matrix(conf_matrix, labels, normalize)

    def align_texts_levenshtein(self, true, prediction):
        """Levenshtein távolság alapú karakterillesztés OCR hibákhoz."""
    
        #allowed_chars = string.ascii_letters
        #true = [c for c in true if c in allowed_chars]
        #prediction = [c for c in prediction if c in allowed_chars]

        true = true.replace("\n", "")
        prediction = prediction.replace("\n", "")

        aligned_original = []
        aligned_ocr = []
    
        ops = Levenshtein.editops(true, prediction)  # OCR műveletek
        original_index, ocr_index = 0, 0  # Karakter indexek
    
        for op, orig_idx, ocr_idx in ops:
            while original_index < orig_idx or ocr_index < ocr_idx:
                aligned_original.append(true[original_index] if original_index < len(true) else "-")
                aligned_ocr.append(prediction[ocr_index] if ocr_index < len(prediction) else "-")
                original_index += 1
                ocr_index += 1
    
            if op == "insert":
                aligned_original.append("-")
                aligned_ocr.append(prediction[ocr_idx])
                ocr_index += 1
    
            elif op == "delete":
                aligned_original.append(true[orig_idx])
                aligned_ocr.append("-")
                original_index += 1
    
            elif op == "replace":
                aligned_original.append(true[orig_idx])
                aligned_ocr.append(prediction[ocr_idx])
                original_index += 1
                ocr_index += 1
    
        while original_index < len(true) or ocr_index < len(prediction):
            aligned_original.append(true[original_index] if original_index < len(true) else "-")
            aligned_ocr.append(prediction[ocr_index] if ocr_index < len(prediction) else "-")
            original_index += 1
            ocr_index += 1
    
        return list(zip(aligned_original, aligned_ocr))
    
    def generate_confusion_matrix(self, labels, true, prediction, normalize):
        aligned_pairs = self.align_texts_levenshtein(true, prediction)
        """Konfúziós mátrix létrehozása az OCR hibákból."""
        confusion_dict = defaultdict(int)
    
        for orig, ocr in aligned_pairs:
            confusion_dict[(orig, ocr)] += 1  # Összesítjük a tévesztéseket
    
        #chars = sorted(set(k for pair in confusion_dict.keys() for k in pair))  # Egyedi karakterek
    
        char_to_idx = {char: i for i, char in enumerate(labels)}  # Indexelés
        matrix_size = len(labels)
        confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
        for (orig, ocr), count in confusion_dict.items():
            confusion_matrix[char_to_idx[orig], char_to_idx[ocr]] = count

        if normalize:
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            confusion_matrix = confusion_matrix / row_sums

    
        return confusion_matrix
    
    def plot_confusion_matrix(self, conf_matrix, labels, normalize, save_path):
        labels = ['space' if x==' ' else x for x in labels]
        """Megjeleníti a konfúziós mátrixot hőtérképként."""
        fig, ax = plt.subplots(figsize=(20,20))
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')

        fmt = ".2f" if normalize else "d"

        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 6})
        plt.yticks(rotation=90)
        plt.xlabel("OCR kimenet")
        plt.ylabel("Eredeti karakter")

        
        plt.title("OCR Konfúziós Mátrix")
        
        path = f"{save_path}/confusion_matrix.png"
        util.create_path(path)
        plt.savefig(path)
        plt.show()



     
     
     
     