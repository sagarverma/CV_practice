import numpy as np
from scipy import signal
import cv2

fin = open('samples/sampleTest1.txt','r')

r,c = map(int, fin.readline().split())

img = []

for line in fin.readlines():
    line_lst = line.split()
    temp = []
    for l in line_lst:
        temp.append(map(int, l.split(',')))
    img.append(temp)

img = np.asarray(img, dtype=np.uint8)

gray_img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

#cv2.imwrite('gray.jpg',gray_img)

#gaussian filter
sig = 1.4
k = 2
gaussian_filter = []

for i in range(1,2*k+1+1):
    temp = []
    for j in range(1,2*k+1+1):
        temp.append((1/(2*np.pi*(sig**2))) * np.exp((-1*((i-(k+1))**2+(j-(k+1))**2))/(2*sig**2)))
    gaussian_filter.append(temp)

gaussian_filter = np.asarray(gaussian_filter)

#print gaussian_filter

smoother_gray_image = signal.convolve2d(gray_img,gaussian_filter,boundary='symm',mode='same')

#cv2.imwrite('smoother_gray_image.jpg',smoother_gray_image)

#Finding the intensity gradient of the image using Sobel operator

x_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]
x_filter = np.asarray(x_filter)
y_filter = x_filter.T

G_x = signal.convolve2d(smoother_gray_image, x_filter, boundary='symm', mode='same')
G_y = signal.convolve2d(smoother_gray_image, y_filter, boundary='symm', mode='same')

#cv2.imwrite('G_x.jpg',G_x)
#cv2.imwrite('G_y.jpg',G_y)

G = np.sqrt(np.square(G_x) + np.square(G_y))
theta = np.arctan2(G_y,G_x)

cv2.imwrite('G.jpg',G)