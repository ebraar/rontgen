import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "edges"
img = cv2.imread("146.jpg", 0)
cv2.imshow("renkli resim",img)

# histogram işlemi
equalized_image = cv2.equalizeHist(img)
cv2.imshow("equalized image", equalized_image)

# renkliden griye
gray_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
cv2.imshow('gray scale image', gray_image)

# 5x5 boyutunda kernel oluşturma
kernel = np.ones((5,5), np.uint8)

#dilation işlemi
img_dilation = cv2.dilate(img, kernel, iterations=1) 
cv2.imshow('dilation image', img_dilation)

# erozyon işlemi
eroded_image = cv2.erode(img_dilation, kernel, iterations=1)
cv2.imshow('eroded image', eroded_image)

# kenar bulma işlemi
edges = cv2.Canny(eroded_image, 100, 200)
cv2.imshow("canny image", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()