import separator.ocr as ocr
import separator.row as row
import separator.character as character

import textdistance
import numpy as np

from separator.binarizer.binarizer_thresh import BinarizerThresh
from separator.cleaner.cleaner import Cleaner
from separator.row_segmentator.row_segmentator import RowSegmentator
from separator.letter_segmentator.letter_segmentator import LetterSegmentator
from separator.resizer.resizer import Resizer
from separator.recognizer.recognizer import Recognizer
from separator.visualizer.visualizer import Visualizer

from network.model import VGG16

input = """
Lorem ipsum dolor sit amet consectetur adipisicing elit. Enim, error! Eos laborum unde esse magni neque quidem mollitia odit pariatur iure sed modi, consequatur ratione autem natus! Officia, sed assumenda! Asperiores cum, illum blanditiis quasi omnis ullam deleniti! Facere, voluptate culpa aliquam consequatur optio, illum quis architecto necessitatibus sint totam consequuntur rerum laudantium, cum sit libero illo corrupti magnam. Placeat? Sed distinctio excepturi esse alias, reiciendis officiis deleniti dolorum, placeat, nesciunt dolorem repudiandae odio beatae numquam ipsum odit. Et asperiores optio dolorem mollitia labore voluptas non nostrum natus quo fugiat? Sunt in pariatur eum esse molestiae recusandae. Similique atque quo eligendi, suscipit inventore iusto illum error perspiciatis nihil veritatis, cupiditate asperiores? 
"""



def main():
    binarizer = BinarizerThresh()
    cleaner = Cleaner()
    row_segmentator = RowSegmentator()
    letter_segmentator = LetterSegmentator()
    resizer = Resizer()
    recognizer = Recognizer(58, "./network/model_weights_58_15_35_45_kisnagybetu.pth", "./network/index_class_mapping.json")
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, "../images/input/input_consolas.png", "../images/")
    
    output = OCR.run()
    OCR.save_rows("rows/row").save_letters("../images/letters/").saveim_bin("binarized_image/im.png")

    print(len(input))
    print(len(output))
    with open("../output/output.txt", "w") as file:
        file.write(output)

    visualizer = Visualizer("../images/output/")
    visualizer.visualize_confusion_matrix(input.replace(" ", ""), output.replace(" ", ""), "confusion_matrix.png", True)

    print("Levenshtein távolság:", textdistance.levenshtein(input.replace(" ", ""), output.replace(" ", "")))



if __name__ == "__main__":
    main()