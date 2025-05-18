from abc import ABC, abstractmethod
from separator.visualizer.base_visualizer import BaseVisualizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import Levenshtein
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from util import util

class Visualizer(BaseVisualizer):
    def __init__(self, save_path, debug):
        self.save_path = save_path
        self.debug = debug
   
    def visualize_confusion_matrix(self, labels, true, prediction, normalize = False):
        # 2. Konfúziós mátrix készítése
        conf_matrix, labels = self.generate_confusion_matrix(labels, true, prediction, True)

        # 3. Megjelenítés
        self.plot_confusion_matrix(conf_matrix, labels, normalize, self.save_path + 'confusion_matrix.png')

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
        #plt.show()

    def plot_metrics_F1_recall_accuracy_precision(self, true, pred):
        aligned = self.align_texts_levenshtein(true, pred)

        # Extract ground truth and prediction characters
        y_true = [t for t, p in aligned if t != "-"]
        y_pred = [p for t, p in aligned if t != "-"]

        # Compute metrics
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        metrics_matrix = np.array([precision, recall, f1, accuracy])
        labels = ["Precision", "Recall", "F1-Score", "Accuracy"]
        self.plot_metrics_together('metrics.png', 'Metrics for whole text', 'Metrics', 'Values', labels, metrics_matrix)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Extracting all precision values (including per-class, macro avg, and weighted avg)
        precision_values = {label: report[label]['precision'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}
        precision_values['\" \"'] = precision_values[' ']
        del precision_values[' ']
        labels = list(precision_values.keys())
        precisions = list(precision_values.values())
        self.plot_metric('precisions.png', 'Precision per Class', 'Classes', 'Precision', labels, precisions)

        # Extracting all precision values (including per-class, macro avg, and weighted avg)
        recall_values = {label: report[label]['recall'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}
        recall_values['\" \"'] = recall_values[' ']
        del recall_values[' ']
        labels = list(recall_values.keys())
        recalls = list(recall_values.values())
        self.plot_metric('recalls.png', 'Recall per Class', 'Classes', 'Recall', labels, recalls)

        # Extracting all precision values (including per-class, macro avg, and weighted avg)
        f1_values = {label: report[label]['f1-score'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}
        f1_values['\" \"'] = f1_values[' ']
        del f1_values[' ']
        labels = list(f1_values.keys())
        f1_scores = list(f1_values.values())
        self.plot_metric('f1_values.png', 'F1-Score per Class', 'Classes', 'F1-Score', labels, f1_scores)


    def plot_metric(self, name, title, xlabel, ylabel, labels, data):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=data, palette='viridis')
        # Add labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Show the plot
        plt.xticks(rotation=0, ha='right')  # Rotate labels if needed
        plt.tight_layout()  # Make sure everything fits
        plt.savefig(self.save_path + '/' + name)
        print(self.save_path + '/' + name)

    def plot_metrics_together(self, name, title, xlabel, ylabel, labels, data):
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=data, palette="viridis")
        # Add labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_ylim([0.8, 1])
        # Show the plot
        plt.xticks(rotation=0, ha='right')  # Rotate labels if needed
        plt.tight_layout()  # Make sure everything fits
        plt.savefig(self.save_path + '/metrics.png')

     
     
     
     