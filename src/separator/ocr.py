from separator import row
import util.util as util
import cv2
import numpy as np

from separator.binarizer.base_binarizer import BaseBinarizer
from separator.cleaner.base_cleaner import BaseCleaner
from separator.row_segmentator.base_row_segmentator import BaseRowSegmentator
from separator.letter_segmentator.base_letter_segmentator import BaseLetterSegmentator
from separator.resizer.base_resizer import BaseResizer
from separator.recognizer.base_recognizer import BaseRecognizer

class ocr:
    def __init__(self, binarizer: BaseBinarizer, cleaner: BaseCleaner, row_separator: BaseRowSegmentator, letter_separator: BaseLetterSegmentator,      resizer: BaseResizer, recognizer: BaseRecognizer, image_path, save_path):
        self.binarizer = binarizer
        self.cleaner = cleaner
        self.row_separator = row_separator
        self.letter_separator = letter_separator
        self.resizer = resizer
        self.recognizer = recognizer

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



        scale = util.calculate_resize_scale(self.rows)
        for row in self.rows:
            for letter in row.letters:
                letter = self.resizer.resize(letter, scale)


        output = ""
        for row in self.rows:
            for letter in row.letters:
                output += self.recognizer.recognize(letter)

                if letter.space_after:
                    output += " "
            
            output += " "

        return output

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