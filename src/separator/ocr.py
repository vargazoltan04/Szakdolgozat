from separator import row
from util import util as util
import cv2
import numpy as np

from separator.binarizer.base_binarizer import BaseBinarizer
from separator.cleaner.base_cleaner import BaseCleaner
from separator.row_segmentator.base_row_segmentator import BaseRowSegmentator
from separator.letter_segmentator.base_letter_segmentator import BaseLetterSegmentator
from separator.resizer.base_resizer import BaseResizer
from separator.recognizer.base_recognizer import BaseRecognizer

class ocr:
    def __init__(self, binarizer: BaseBinarizer, cleaner: BaseCleaner, row_separator: BaseRowSegmentator, letter_separator: BaseLetterSegmentator, resizer: BaseResizer, recognizer: BaseRecognizer, image_path, save_path):
        self.binarizer = binarizer
        self.cleaner = cleaner
        self.row_separator = row_separator
        self.letter_separator = letter_separator
        self.resizer = resizer
        self.recognizer = recognizer

        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.save_path = save_path
        self.rows: row = []
        self.output = ""

    def run(self):
        self.image = self.binarizer.binarize(self.image, 128)
        self.image = util.delete_small_components(self.image, 5)
        #self.rows, row_lines_red = self.row_separator.row_segmentation(self.image)
        self.rows, rows_rect_image, rows_dilated, masks = self.row_separator.row_segmentation(self.image)
        for i in range(len(self.rows)):
            self.rows[i].letters, letter_lines_red = self.letter_separator.letter_segmentation(self.rows[i])
            self.saveim(letter_lines_red, f"/rows_lined/row_lined{i}.png")

        scale = util.calculate_resize_scale(self.rows, self.resizer.target_char_size)
        for row in self.rows:
            for letter in row.letters:
                letter = self.resizer.resize(letter, scale)


        for row in self.rows:
            for letter in row.letters:
                self.output += self.recognizer.recognize(letter)

                if letter.space_after:
                    self.output += " "
            
            self.output += " "

        self.save_rows(f"{self.save_path}/rows")
        self.save_letters(f"{self.save_path}/letters")
        self.save_output("output.txt")
        #self.saveim(row_lines_red, "row_lines_red.png")
        self.saveim(rows_rect_image, "rows_bounding_rects.png")
        self.saveim(rows_dilated, "rows_dilated.png")
        self.saveim(masks[1], "row_mask.png")

        return self

    def show(self, window_name):
        cv2.imshow(window_name, self.image)
        return self
    
    def saveim(self, image, file_name):
        path = f"{self.save_path}/{file_name}"
        util.create_path(path)
        cv2.imwrite(path, image)
        return self
    
    def save_rows(self, file_name):
        for i in range(len(self.rows)):
            self.rows[i].save_row(file_name)

        return self
    
    def save_letters(self, file_name):
        for i in range(len(self.rows)):
            self.rows[i].save_letters(file_name)
        
        return self
    
    def get_output(self):
        return self.output
    
    def save_output(self, file_name):
        path = f"{self.save_path}/{file_name}"
        util.create_path(path)
        with open(path, "w") as file:
            file.write(self.output)