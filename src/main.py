import separator.ocr as ocr
import separator.row as row
import separator.character as character

import textdistance
import numpy as np

from separator import *


from network.model import VGG16

input = """
Lorem ipsum dolor sit amet consectetur adipisicing elit. Enim, error! Eos laborum unde esse magni neque quidem mollitia odit pariatur iure sed modi, consequatur ratione autem natus! Officia, sed assumenda! Asperiores cum, illum blanditiis quasi omnis ullam deleniti! Facere, voluptate culpa aliquam consequatur optio, illum quis architecto necessitatibus sint totam consequuntur rerum laudantium, cum sit libero illo corrupti magnam. Placeat? Sed distinctio excepturi esse alias, reiciendis officiis deleniti dolorum, placeat, nesciunt dolorem repudiandae odio beatae numquam ipsum odit. Et asperiores optio dolorem mollitia labore voluptas non nostrum natus quo fugiat? Sunt in pariatur eum esse molestiae recusandae. Similique atque quo eligendi, suscipit inventore iusto illum error perspiciatis nihil veritatis, cupiditate asperiores? 
"""



def main():
    binarizer: BaseBinarizer = BinarizerThresh()
    cleaner: BaseCleaner = Cleaner()
    row_segmentator: BaseRowSegmentator = RowSegmentator()
    letter_segmentator: BaseLetterSegmentator = LetterSegmentator()
    resizer: BaseResizer = Resizer(45)
    recognizer: BaseRecognizer = Recognizer(58, "./network/model_weights_58_15_36_51_szurke.pth", "./network/index_class_mapping.json")
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, "../images/input/input_consolas.png", "../images/")
    
    output = OCR.run()
    OCR.save_rows("rows/row").save_letters("../images/letters/").saveim_bin("binarized_image/im.png")

    print(len(input))
    print(len(output))
    with open("../output/output.txt", "w") as file:
        file.write(output)


   
    visualizer: BaseVisualizer = Visualizer("../images/output/")
    #visualizer.visualize_confusion_matrix(input.replace(" ", ""), output.replace(" ", ""), "confusion_matrix.png", True)
    #visualizer.visualize_confusion_matrix(input.replace(" ", "").lower(), output.replace(" ", "").lower(), "confusion_matrix.png", True)
    visualizer.visualize_confusion_matrix(input, output, True)
    

    print(output)
    print("Levenshtein távolság:", textdistance.levenshtein(input.replace(" ", ""), output.replace(" ", "")))



if __name__ == "__main__":
    main()