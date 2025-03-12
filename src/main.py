import separator.ocr as ocr
import separator.row as row
import separator.character as character

import textdistance
import numpy as np

from separator import *


from network.model import VGG16

input = """
Lorem ipsum dolor sit amet consectetur adipisicing elit. Numquam ad nulla repellendus maiores, officia alias tenetur, inventore quae quaerat nesciunt iste omnis, amet a esse at reiciendis fugit iure commodi. Veniam voluptatem provident vero ullam iste fugit, laboriosam numquam vel eos, repellendus placeat facilis deleniti, animi laudantium nam perspiciatis aspernatur. Tenetur et consectetur ad nihil nulla sunt quos, esse vel. Dolores amet voluptas adipisci fugiat ut ab rerum repudiandae nemo aspernatur inventore ullam a vitae, ducimus illo qui ea, illum neque? Maiores, doloremque sint vero odio consectetur hic rerum rem! Dicta libero illum quos, quam dolor dignissimos fugiat eum aliquid ipsa quidem aperiam unde eius qui excepturi iste magnam alias perferendis, architecto illo voluptatibus dolores! Nam illo voluptas beatae nobis? Voluptatibus recusandae voluptates ipsum. Voluptatum repellat blanditiis dolor quod, voluptatem dolores quo reiciendis modi labore perspiciatis eligendi dolorum quasi quia? Dolorem adipisci officia omnis eum quos neque ea, recusandae minus! Eos, saepe dicta rerum eveniet doloribus, quibusdam laboriosam consequuntur possimus iste veniam pariatur assumenda, temporibus ex ipsa id delectus? Necessitatibus ducimus obcaecati nihil sed sunt, incidunt vitae eaque blanditiis possimus. Deserunt impedit suscipit repellendus, est quo, dolor optio sequi, mollitia quasi fugit asperiores harum nesciunt ea quidem neque rerum delectus earum exercitationem ipsam. Repellendus consequatur id ipsam sapiente ea ex. Autem reprehenderit in cumque aliquid excepturi, nam magni accusamus molestiae reiciendis eos ex veniam delectus totam consectetur vero mollitia dicta sit. Beatae, deserunt! Saepe quibusdam corrupti aspernatur expedita autem accusantium. Nesciunt eligendi ex mollitia similique ipsum, et itaque sint laboriosam, ut sapiente cum ipsam ullam reprehenderit vero eos culpa facere debitis omnis ad aliquam! Repellat sunt delectus ea! Aspernatur, nostrum. Quasi, est. Voluptates, quia neque illo cupiditate, necessitatibus beatae rem optio adipisci iste porro dolor voluptatem hic tempore obcaecati sapiente quos similique quod mollitia. Nulla sint deserunt pariatur qui assumenda.
"""



def main():
    binarizer = BinarizerThresh()
    cleaner = Cleaner()
    row_segmentator = RowSegmentator()
    letter_segmentator = LetterSegmentator()
    resizer = Resizer()
    recognizer = Recognizer(58, "./network/model_weights_58_15-35-55_kisnagybetu.pth", "./network/index_class_mapping.json")
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, "../images/input/input_01.png", "../images/")
    binarizer: BaseBinarizer = BinarizerThresh()
    cleaner: BaseCleaner = Cleaner()
    row_segmentator: BaseRowSegmentator = RowSegmentator()
    letter_segmentator: BaseLetterSegmentator = LetterSegmentator()
    resizer: BaseResizer = Resizer(45)
    recognizer: BaseRecognizer = Recognizer(58, "./network/model_weights_58_15_36_51_szurke.pth", "./network/index_class_mapping.json")
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, "../images/input/input_consolas.png", "../images/")
    
    output = OCR.run()
    OCR.save_rows("rows/row").save_letters("../images/letters/")

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
    visualizer = Visualizer("../images/output/")
    visualizer.visualize_confusion_matrix(input, output, "confusion_matrix.png", True)

    
    print("Levenshtein távolság:", textdistance.levenshtein(input.replace(" ", ""), output.replace(" ", "")))



if __name__ == "__main__":
    main()