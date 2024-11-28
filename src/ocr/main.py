import separator.ocr as ocr
import cv2

def main():
    OCR = ocr.ocr("../../images/input/test01.jpg", "../images/")

    OCR.binarize().delete_small_components(10).row_segmentation().save_rows("rows/row").letter_segmentation().save_letters("../images/letters/")
    #OCR.binarize().delete_small_components(10).row_segmentation().letter_segmentation()

    cv2.waitKey(0)

if __name__ == "__main__":
    main()