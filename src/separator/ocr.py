from separator import row
import util.util as util
import cv2
import numpy as np

class ocr:
    def __init__(self, binarizer, cleaner, row_separator, letter_separator, image_path, save_path):
        self.binarizer = binarizer
        self.cleaner = cleaner
        self.row_separator = row_separator
        self.letter_separator = letter_separator

        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.bin_image = None
        self.save_path = save_path
        self.rows: row = []

    def run(self):
        self.bin_image = self.binarizer.binarize(self.image, 128)
        self.image, self.bin_image = self.cleaner.delete_small_components(self.image, self.bin_image, 5)
        self.rows = self.row_separator.row_segmentation(self.image, self.bin_image)

        for row in self.rows:
            row.letters = self.letter_separator.letter_segmentation(row)
        return self

    def show(self, windowName):
        cv2.imshow(windowName, self.image)
        return self
    
    def saveim(self, filename):
        cv2.imwrite(self.save_path + filename, self.image)
        return self
    
    def saveim_bin(self, filename):
        cv2.imwrite(self.save_path + filename, self.bin_image)
        return self
    
    def save_rows(self, filename):
        for i in range(len(self.rows)):
            self.rows[i].save_row(self.save_path + filename)

        return self
    
    def save_letters(self, filename):
        for i in range(len(self.rows)):
            self.rows[i].save_letters(filename)
        
        return self
    
    def resize(self):
        min_scale = float('inf')
        for row in self.rows:
            for letter in row.letters:
                original_height, original_width = letter.char.shape

                if original_width == 0 or original_height == 0:
                    print(f"Figyelmeztetés: Üres betű észlelve! Kihagyva. ({original_width}x{original_height})")
                    continue  # Kihagyjuk ezt a betűt

                scale = min(45 / original_width, 45 / original_height)
                if scale < min_scale:
                    min_scale = scale

        for row in self.rows:
            for letter in row.letters:
                if letter.char.shape[0] == 0 or letter.char.shape[1] == 0:
                    continue
                
                letter.resize(min_scale)

        return self