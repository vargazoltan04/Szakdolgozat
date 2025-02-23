import separator.ocr as ocr
import separator.row as row
import separator.character as character
import cv2
import torch
import PIL.Image as Image
import PIL
import json
import numpy as np

from separator.binarizer.binarizer_thresh import BinarizerThresh
from separator.cleaner.cleaner import Cleaner
from separator.row_segmentator.row_segmentator import RowSegmentator
from separator.letter_segmentator.letter_segmentator import LetterSegmentator

from network.model import VGG16
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),       # Resize images to 64x64
      # Convert images to grayscale (single channel)
    transforms.ToTensor(),             # Convert images to tensor
])


def main():
    binarizer = BinarizerThresh()
    cleaner = Cleaner()
    row_segmentator = RowSegmentator()
    letter_segmentator = LetterSegmentator()
    OCR = ocr.ocr(binarizer, cleaner, row_segmentator, letter_segmentator, "../images/input/test01.png", "../images/")

    OCR.run().save_rows("rows/row").resize().save_letters("../images/letters/").saveim_bin("binarized_image/im.png")
    #OCR.binarize().delete_small_components(10).row_segmentation().letter_segmentation() 

    model = VGG16(58) 
    model.load_state_dict(torch.load("./network/model_weights.pth", weights_only=True, map_location=torch.device('cpu')))

    # Load the JSON data into a dictionary
    with open('./network/index_class_mapping.json', 'r') as json_file:
        index_class_mapping = json.load(json_file)

    #im = Image.open("../train_data/class/b/Abadi_2.png")
    output = ""
    model.eval()
    for line in OCR.rows:
        for letter in line.letters:
            _, predicted_class = torch.max(model(transform(Image.fromarray(letter.char)).unsqueeze(0)), 1)
            output += index_class_mapping[str(predicted_class.item())]

            if letter.space_after:
                output += " "

        output += "\n"

    print(output)

    with open("../output/output.txt", "w") as file:
        file.write(output)




if __name__ == "__main__":
    main()