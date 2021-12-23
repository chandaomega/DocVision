import numpy as np
import cv2
import os
import csv
import glob
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.model_selection import train_test_split
from joblib import load,dump
import sklearn.metrics as sm

def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))
    return(sm.accuracy_score(label_test,pred), sm.precision_score(label_test,pred,average='micro'),sm.f1_score(label_test,pred,average='micro'),sm.recall_score(label_test,pred,average='micro'))

def predict_svm(X_train, X_test, y_train, y_test):
    svc=svm.SVC(kernel='linear') 
    print("svm started")
    svc.fit(X_train,y_train)
    dump(svc, 'svm.model')
    y_pred=svc.predict(X_test)
    return calc_accuracy("SVM",y_test,y_pred)

X = []
label = []
for j in [0,1,2]:
    files = glob.glob("D:\\Drive\\projects\\computer science\\python\\Retinology\\Trains\\"+str(j)+"/*.jpg")
    for i in files:
        print(i)
        img = cv2.imread(i)
        img = cv2.resize(img,(128,128))
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((10,10),np.float32)/64
        st = cv2.filter2D(gry,-1,kernel)
        Avg = cv2.blur(st,(8,8))
        thresh = 80
        img_th = cv2.threshold(Avg, thresh, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        erodeim = cv2.erode(img_th,kernel,iterations = 1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        DilImg = cv2.dilate(erodeim,kernel2,iterations = 1)
        edges = cv2.Canny(DilImg,100,200)
        cv2.imshow('Image', gry)
        cv2.waitKey(100)
        fet=[]
        glcm = greycomatrix(gry, [5], [0], 256, symmetric=True, normed=True)
        for ii in (greycoprops(glcm, 'dissimilarity')):
            for jj in ii:
                fet.append(jj)
        for ii in (greycoprops(glcm, 'correlation')):
            for jj in ii:
                fet.append(jj)            
        label.append(j)
        X.append(fet)
    
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.33, random_state=42)
print(predict_svm(X_train, X_test, y_train, y_test))
