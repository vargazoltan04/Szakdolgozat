import cv2
import ocr

OCR = ocr.ocr("../images/input/test01.png", "../images/")

OCR.binarize().delete_small_components(10).row_segmentation().save_rows("rows/row").letter_segmentation().save_letters("../images/letters/")
#OCR.binarize().delete_small_components(10).row_segmentation().letter_segmentation()

cv2.waitKey(0)