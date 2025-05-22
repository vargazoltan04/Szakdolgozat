import separator.ocr as ocr
import separator.row as row
import separator.character as character

import textdistance
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


from separator import *
from separator.row_segmentator.row_segmentator_new import RowSegmentatorNew

from util import util

from network.model import VGG16

def main(path, output_path, debug, debug_txt):
    print(path, output_path, debug, debug_txt)
    recognizer = Recognizer(58, "./network/model_weights_58_15-35-55_kisnagybetu.pth", "./network/index_class_mapping.json", debug)
    binarizer = BinarizerThresh(debug)
    cleaner = Cleaner(debug)
    row_segmentator = RowSegmentatorNew(debug)
    letter_segmentator = LetterSegmentator(debug)
    resizer = Resizer(45, debug)
    visualizer: BaseVisualizer = Visualizer(output_path, debug)

    if debug:
        print(debug_txt)
        debug_file = open(debug_txt)
        input4 = debug_file.read()
        debug_file.close()
    
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, path, output_path, debug)
    OCR.run()
    output = OCR.get_output()
    
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,-: "
    if debug:
        cm = visualizer.generate_confusion_matrix(labels, input4, output, True)
        visualizer.plot_confusion_matrix(cm, labels, True, output_path)
        visualizer.plot_metrics_F1_recall_accuracy_precision(input4, output)

        print("Levenshtein távolság:", textdistance.levenshtein(input4, OCR.get_output()))
        print("Hasonlósági arány: ", util.char_by_char_similarity(visualizer.align_texts_levenshtein(input4, OCR.get_output())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help = "The path of the image")
    parser.add_argument("-op", "--output_path", help = "The output path of the result")
    parser.add_argument('--debug', action='store_true', help='Debug mód bekapcsolása')
    args = parser.parse_args()

    path = args.path
    path_obj = Path(args.path)
    output_path = args.output_path
    debug = args.debug

    if output_path is None:
        output_path = str(path_obj.parent)
        util.create_path(output_path)

    print("The input image: %s" % path)
    print("The output path: %s" % output_path)

    if not path_obj.suffix == ".png" and not path_obj.suffix == ".jpg":
        print("Give me an input image!")
    else:
        debug_txt = str(path_obj.parent) + "\\" + path_obj.stem + ".txt"
        main(args.path, output_path, args.debug, debug_txt)