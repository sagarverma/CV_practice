import numpy as np
from scipy import signal, ndimage
import cv2
from math import ceil, floor, cos, sin

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

cv2.imwrite('color.jpg',img)

gray_img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

cv2.imwrite('gray.jpg',gray_img)

#gaussian filter
sig = 3.0
ker = 5
gaussian_filter = []

for i in range(1,2*ker+1+1):
    temp = []
    for j in range(1,2*ker+1+1):
        temp.append((1/(2*np.pi*(sig**2))) * np.exp((-1*((i-(ker+1))**2+(j-(ker+1))**2))/(2*sig**2)))
    gaussian_filter.append(temp)

gaussian_filter = np.asarray(gaussian_filter)

#print gaussian_filter

smoother_gray_image = signal.convolve2d(gray_img,gaussian_filter,boundary='symm',mode='same')

#print smoother_gray_image.shape
#print img.shape

cv2.imwrite('smoother_gray_image.jpg',smoother_gray_image)

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
theta = (theta * 180 / np.pi) % 360

for i in range(r):
    for j in range(c):
        t = theta[i][j]
        if (t < 22.5 and t >= 0) or (t >= 157.5 and t < 202.5) or (t >= 337.5 and t <= 360):
            t = 0
        elif (t >= 22.5 and t < 67.5) or (t >= 202.5 and t < 247.5):
            t = 45
        elif (t >= 67.5 and t < 112.5) or (t >= 247.5 and t < 292.5):
            t = 90
        else:
            t = 135
        theta[i][j] = t

cv2.imwrite('G.jpg',G)
cv2.imwrite('theta.jpg',theta)

#Non Maximum Suppression

non_max_supp = G.copy()

for i in range(1,r-1):
    for j in range(1,c-1):
        if theta[i][j] == 0:
            if (G[i][j] <= G[i][j+1]) or (G[i][j] <= G[i][j-1]):
                non_max_supp[i][j] = 0
        elif theta[i][j] == 45:
            if (G[i][j] <= G[i+1][j+1]) or (G[i][j] <= G[i+1][j-1]):
                non_max_supp[i][j] = 0
        elif theta[i][j] == 90:
            if (G[i][j] <= G[i+1][j]) or (G[i][j] <= G[i-1][j]):
                non_max_supp[i][j] = 0
        else:
            if (G[i][j] <= G[i+1][j+1]) or (G[i][j] <= G[i-1][j-1]):
                non_max_supp[i][j] = 0 

cv2.imwrite('non_max_supp.jpg',non_max_supp)


#Edge tracking by hysteresis

maxx = np.max(non_max_supp)
th = 0.2*maxx
tl = 0.1*maxx

gnh = np.zeros((r,c))
gnl = np.zeros((r,c))

for i in range(r):
    for j in range(c):
        if non_max_supp[i][j] >= th:
            gnh[i][j] = non_max_supp[i][j]
        if non_max_supp[i][j] >= tl:
            gnl[i][j] = non_max_supp[i][j]

gnl = gnl-gnh

cv2.imwrite('gnl.jpg',gnl)
cv2.imwrite('gnh.jpg',gnh)

def traverse(i, j):
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    for k in range(8):
        if gnh[i+x[k]][j+y[k]]==0 and gnl[i+x[k]][j+y[k]]!=0:
            gnh[i+x[k]][j+y[k]]=255
            traverse(i+x[k], j+y[k])


for i in range(1, r-10):
    for j in range(1, c-10):
        if gnh[i][j]:
            gnh[i][j]=255
            traverse(i, j)

cv2.imwrite('canney_edge_detected.jpg',gnh)


#Hough Transform
dmin = int(ceil(-1 * max(r,c) * np.sqrt(2)))
dmax = -1 * dmin
thetamin = 0
thetamax = 90

#print dmin, dmax

thetabins = (thetamax - thetamin + 1) / 1
dbins = int(ceil((dmax - dmin + 1) / 2))

accumulator = np.zeros((dbins,thetabins))

#print accumulator.shape

for i in range(r):
    for j in range(c):
        if gnh[i][j]:
            for theta_j in range(91):
                d = i * cos(theta_j * np.pi / 180) - j * sin(theta_j * np.pi / 180)
                #print d
                d = int(ceil((d + dmax)/2))
                #print d
                accumulator[d ,theta_j] += 1

cv2.imwrite('accumulator.jpg',accumulator)


