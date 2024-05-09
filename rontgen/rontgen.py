import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img_path = 'rontgen/146.jpg'
img = cv2.imread(img_path, 0)
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
img_dilation = cv2.dilate(gray_image, kernel, iterations=1) 
cv2.imshow('dilation image', img_dilation)

# erozyon işlemi
eroded_image = cv2.erode(img_dilation, kernel, iterations=1)
cv2.imshow('eroded image', eroded_image)

# Closing işlemi
closing = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

# Opening işlemi
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# kenar bulma işlemi
edges = cv2.Canny(closing, 100, 200)
cv2.imshow("canny image", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()