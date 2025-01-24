import separator.ocr as ocr
import separator.row as row
import separator.character as character
import cv2
import torch
import PIL.Image as Image
import PIL
import json
import numpy as np

from network.model import VGG16
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),       # Resize images to 64x64
      # Convert images to grayscale (single channel)
    transforms.ToTensor(),             # Convert images to tensor
])


def main():
    OCR = ocr.ocr("../images/input/input.png", "../images/")

    OCR.binarize().saveim_bin("binarized_image/im.png").delete_small_components(10).row_segmentation().save_rows("rows/row").letter_segmentation().save_letters("../images/letters/")
    #OCR.binarize().delete_small_components(10).row_segmentation().letter_segmentation() 

    model = VGG16(52)
    model.load_state_dict(torch.load("./network/model_weights.pth", weights_only=True))

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

        output += "\n"

    print(output)
if __name__ == "__main__":
    main()