import cv2
import numpy as np

image = cv2.imread("../images/input/test01.png")
cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.imshow("input", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_inverted = cv2.bitwise_not(image_gray)
_, image_binary = cv2.threshold(image_inverted, 128, 255, cv2.THRESH_BINARY)

num_labels, labels = cv2.connectedComponents(image_binary)

sizes = np.bincount(labels.ravel())
min_size = 10
for label in range(0, num_labels):
    if sizes[label] < min_size:
        labels[labels == label] = 0

binary_output = np.where(labels > 0, 255, 0).astype(np.uint8)
binary_output = cv2.bitwise_not(binary_output)
cv2.imshow('label', binary_output)

cv2.namedWindow("BINARY", cv2.WINDOW_NORMAL)
cv2.imwrite("../images/binarized_image/test_binary.png", binary_output)
cv2.waitKey(0)