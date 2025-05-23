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
import sys



def main(path, output_path, debug, debug_txt):
    #Modulok létrehozása
    recognizer = Recognizer(58, "./network/model_weights_final.pth", "./network/index_class_mapping.json", debug)
    binarizer = BinarizerThresh(debug)
    cleaner = Cleaner(debug)
    row_segmentator = RowSegmentatorNew(debug)
    letter_segmentator = LetterSegmentator(debug)
    resizer = Resizer(45, debug)
    visualizer: BaseVisualizer = Visualizer(output_path, debug)

    #Ha debug mód, akkor a ground_truth szöveg olvasása
    if debug:
        debug_file = open(debug_txt)
        input4 = debug_file.read()
        debug_file.close()
    
    #Program futtatása
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, resizer, recognizer, path, output_path, debug)
    OCR.run()
    output = OCR.get_output()
    
    #Ha debug mód, akkor az ábrák mentése
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,-: "
    if debug:
        cm = visualizer.generate_confusion_matrix(labels, input4, output, True)
        visualizer.plot_confusion_matrix(cm, labels, True, output_path)
        visualizer.plot_metrics_F1_recall_accuracy_precision(input4, output)

        print("Levenshtein távolság:", textdistance.levenshtein(input4, OCR.get_output()))
        print("Hasonlósági arány: ", util.char_by_char_similarity(visualizer.align_texts_levenshtein(input4, OCR.get_output())))

if __name__ == "__main__":
    sys.tracebacklimit = 0

    #Parancssori paramétereket hozza létre
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help = "The path of the image")
    parser.add_argument("-op", "--output_path", help = "The output path of the result")
    parser.add_argument('--debug', action='store_true', help='Turn on debug mode')
    args = parser.parse_args()

    
    #Parancssori paraméterek beolvasása, és program futtatása megfelelően
    path = ""
    if args.path:
        path = args.path
    else:
        raise Exception("You have to give me a path to the input file")

    path_obj = Path(args.path)
    output_path = args.output_path
    debug = args.debug

    if output_path is None:
        output_path = str(path_obj.parent)
        util.create_path(output_path)

    print("The input image: %s" % path)
    print("The output path: %s" % output_path)

    if not path_obj.suffix == ".png" and not path_obj.suffix == ".jpg":
        raise Exception("You have to give me a path to a .png or a .jpg file")
    else:
        debug_txt = str(path_obj.parent) + "\\" + path_obj.stem + ".txt"
        try:
            main(args.path, output_path, args.debug, debug_txt)
        except Exception as ex:
            print(ex)