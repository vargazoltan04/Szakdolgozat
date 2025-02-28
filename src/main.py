import separator.ocr as ocr
import separator.row as row
import separator.character as character

import numpy as np

from separator.binarizer.binarizer_thresh import BinarizerThresh
from separator.cleaner.cleaner import Cleaner
from separator.row_segmentator.row_segmentator import RowSegmentator
from separator.letter_segmentator.letter_segmentator import LetterSegmentator
from separator.resizer.resizer import Resizer
from separator.recognizer.recognizer import Recognizer

from network.model import VGG16





def main():
    binarizer = BinarizerThresh()
    cleaner = Cleaner()
    row_segmentator = RowSegmentator()
    letter_segmentator = LetterSegmentator()
    resizer = Resizer()
    recognizer = Recognizer(58, "./network/model_weights_58_15_30_50_kisnagybetu_szurkearnyalat.pth", "./network/index_class_mapping.json")
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, "../images/input/test01.png", "../images/")
    
    output = OCR.run()
    OCR.save_rows("rows/row").save_letters("../images/letters/").saveim_bin("binarized_image/im.png")

    print(output)
    with open("../output/output.txt", "w") as file:
        file.write(output)




if __name__ == "__main__":
    main()