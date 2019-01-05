import cv2
import numpy as np
import os

path='/home/ganesh/hough'
#path='drive/Colab'
#os.remove("drive/Colab/book_hough2.jpg")

img = cv2.imread(path+'/book.jpg')
#ret,gray = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#cv2.imwrite(path+'/book_hough2.jpg',edges)
cv2.imshow('output',img);
cv2.waitKey(0);
print "end"


'''
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 0
maxLineGap = 0
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
lines = cv2.HoughLinesP(gray,1,np.pi/180*100,200,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


'''
