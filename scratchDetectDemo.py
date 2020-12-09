import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define Upper-Lower Bound of threshold
threshval = (80,255)

# normaize the grayscale image
# NO LONGER SIGNIFICANT, ONLY USE WHEN CREATING CUSTOM FILTERS
def nm(img):
	normalizedImg = np.zeros(img.shape)
	return cv2.normalize(img, normalizedImg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Image Resizing function
def imgresize(timg, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)

	# resize image
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


# Read File name
file = sys.argv[1]


print('Reading from file: ',file)


# Show steps
# 1 is true, 0 is false
disp = 1

# Read an image
img = cv2.imread(file, 0)
imgorig = cv2.imread(file)

# Resize the image
gray = imgresize(img, 50)
imgorig = imgresize(imgorig, 50)

# Converting grayscale to RGB for contour visualization 
imgorig = cv2.merge([imgorig,imgorig,imgorig])


# Show both images
if disp == 1:
	cv2.imshow('Original image',img)
	cv2.imshow('Gray image', gray)

# Apply a Gaussian filter of size 15x15
gSize = 5;

# gray = imfilter(gray,fspecial('gaussian',[gSize,gSize],gSize/2),'replicate')
blur = cv2.GaussianBlur(gray,(gSize,gSize),0)

# Diplay Blur
if disp == 1:
	cv2.imshow('Gaussian Filter', blur)


# Calulcate Laplacian for reference (absolute value, 1 or 0)
# NOT TO BE USED DUE TO LACK OF ACCURACY
laplacian = cv2.Laplacian(blur,cv2.CV_64F, ksize=13) 


# Sobel Filter
sobelH = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float)
sobelV = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float)

# Calculate Gradient Magnitude for image

# sh = cv2.filter2D(gray, -1, sobelH)
# sv = cv2.filter2D(gray, -1, sobelV)
# sm = np.sqrt(sh**2 + sv**2).astype(np.uint8)

dx = cv2.Sobel(blur, cv2.CV_64F,1,0,sobelH)
dy = cv2.Sobel(blur, cv2.CV_64F,0,1,sobelV)
axy = cv2.magnitude(dx, dy).astype(np.uint8)


# Display Stuff
if disp == 1:
	cv2.imshow('Laplacian (FOR REFERENCE)', laplacian)
#	cv2.imshow('Sobel X Mask', sobelx)
#	cv2.imshow('Sobel Y Mask', sobely)
	cv2.imshow('Gradient Magnitude (Sobel Mask)', axy)

# Threshhold image
ret1, thresh = cv2.threshold(axy,threshval[0],threshval[1],cv2.THRESH_BINARY)
# threshadapt = cv2.adaptiveThreshold(axy,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# threshmean = cv2.adaptiveThreshold(axy,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)


cv2.imshow('Threshold', thresh)
# cv2.imshow('test2',threshadapt)
# cv2.imshow('test3', threshmean)


# Morphology to determine contours
kernel = np.ones((1,1),np.uint8)
ret = cv2.erode(thresh,kernel,iterations = 2)


kernel = np.ones((1,1),np.uint8)
# dilate = cv2.dilate(thresh,kernel,iterations = 1)
# erosion = cv2.erode(dilate,kernel,iterations = 1)
ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((5,5),np.uint8)
ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((2,2),np.uint8)
ret = cv2.dilate(ret,kernel,iterations = 4)

cv2.imshow('Morphology', ret)


# Calculate Contours
contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgorig, contours, -1, (0,255,0), 1)

cv2.imshow('Contours', imgorig)

cv2.waitKey()
