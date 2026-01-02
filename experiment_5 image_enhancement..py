import cv2
import numpy as np
# Read the image
image = cv2.imread(r"C:\Users\Asus\Desktop\practical\WhatsApp Image 2025-12-23 at 22.37.11.jpeg")
# Check if image is loaded
if image is None:
    print("Error: Image not found")
    exit()
# ------------------- Smoothing Filters -------------------
# Using Gaussian Blur
smoothed_image = cv2.GaussianBlur(image, (7,7), 0)
# ------------------- Sharpening Filter -------------------
# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
# ------------------- Display Images -------------------
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image", smoothed_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()