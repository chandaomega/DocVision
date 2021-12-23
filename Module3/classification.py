from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
import os
import csv
import sklearn.metrics as sm
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import random
import warnings
import pickle
import numpy as np
import sklearn.metrics as sm
from joblib import dump, load
import numpy as np
import cv2
import os
import csv
from skimage import morphology
from skimage.feature import hog
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def bloodVessel(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(96,96))
    curImg = img[:,:,1]
    histEqImg= cv2.equalizeHist(curImg)
    gray = curImg
    kernelG1 = np.array([[ 5,  5,  5],[-3,  0, -3],[-3, -3, -3]])
    kernelG2 = np.array([[ 5,  5, -3],[ 5,  0, -3],[-3, -3, -3]])
    kernelG3 = np.array([[ 5, -3, -3],[ 5,  0, -3],[ 5, -3, -3]])
    kernelG4 = np.array([[-3, -3, -3],[ 5,  0, -3],[ 5,  5, -3]])
    kernelG5 = np.array([[-3, -3, -3],[-3,  0, -3],[ 5,  5,  5]])
    kernelG6 = np.array([[-3, -3, -3],[-3,  0,  5],[-3,  5,  5]])
    kernelG7 = np.array([[-3, -3,  5],[-3,  0,  5],[-3, -3,  5]])
    kernelG8 = np.array([[-3,  5,  5],[-3,  0,  5],[-3, -3, -3]]) 
    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    magn = cv2.max(g1, cv2.max(g2, cv2.max(g3, cv2.max(g4, cv2.max(g5, cv2.max(g6, cv2.max(g7, g8)))))))
    curImg = magn
    threshImg = cv2.threshold(curImg,160,180,cv2.THRESH_BINARY_INV)
    cleanImg = morphology.remove_small_objects(curImg, min_size=130, connectivity=100)
    return cleanImg

def Exudates(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(96,96))
    curImg = img[:,:,1]
    histEqImg= cv2.equalizeHist(curImg)
    gray = curImg
    clahe = cv2.createCLAHE()
    clImg = clahe.apply(curImg)
    strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    dilateImg = cv2.dilate(curImg, strEl)
    retValue, threshImg = cv2.threshold(dilateImg, 220, 220, cv2.THRESH_BINARY)
    medianImg = cv2.medianBlur(dilateImg,5)
    return medianImg
    
def Mask(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(96,96))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    eyeMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    eyeMask = cv2.erode(eyeMask, kernel, iterations = 2)
    eyeMask = cv2.dilate(eyeMask, kernel, iterations = 2)
    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")
    eyeMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    eyeMask = cv2.addWeighted(eyeMask,0.5,eyeMask2,0.5,0.0)
    eyeMask = cv2.medianBlur(eyeMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = eyeMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = eyeMask)
    h,w = skin.shape[:2]
    bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.GaussianBlur(bw_image,(5,5),0)
    threshold = 1
    for i in range(h):
        for j in range(w):
            if bw_image[i][j] > threshold:
               bw_image[i][j] = 0
            else:
               bw_image[i][j] = 255
    return bw_image

def main(path):
    fdi=[]
    feature = []
    imr = cv2.imread(path)
    bw_image=Mask(path)
    bvimage=bloodVessel(path)
    Eximage=Exudates(path)
    fdb,hog_image = hog(bvimage, orientations=8, pixels_per_cell=(16,16),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
    fdE,hog_image = hog(Eximage, orientations=8, pixels_per_cell=(16,16),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
    for i in fdb:
        fdi.append(i)
    for j in fdE:
        fdi.append(j)
    feature.append(fdi)
    clf = load('RF.Model')
    val = clf.predict(feature)
    print(val[0])
    if val[0] ==1:
        v = 'Diabetic Ret'
    elif val[0] ==2:
        v='Gulocomo'
    else:
        v='Healthy'
    #cv2.imshow('Image Detected is '+v + '  ',imr)
    return v
