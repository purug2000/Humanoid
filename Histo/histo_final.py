from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import copy
import os
bins=25
kernel = np.ones((5,5),np.uint8)
target = cv.imread('3.jpg')


def find_max_cnt(thresh):
    _,contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
                

        res = contours[ci]
    return res


##Function to find the threshholded image of target, of region similar to comp_image
def find_thresh(comp_img,target,bins):
    ## [Transform it to HSV]
    hsv = cv.cvtColor(comp_img, cv.COLOR_BGR2HSV)
    ## [Transform it to HSV]
    ## [Use only the Hue value]
    ch = (0, 0)
    hue = np.empty(hsv.shape, hsv.dtype)
    cv.mixChannels([hsv], [hue], ch)
    bj=Hist_and_Backproj(bins,target,hue)
    cv.imshow('bj',bj)
    s=[]
    for x in bj:
        s=list(set(s+list(set(x))))
    print(s)
    ret, thresh = cv.threshold(bj, 250, 255, cv.THRESH_BINARY)

    return thresh


##Fuction to take in hue values and back project them to a target image and return the back projection 

def Hist_and_Backproj(val,target,hue):
    ## [initialize]
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range
    ## [initialize]
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    ch = (0, 0)
    hue2 = np.empty(hsvt.shape, hsvt.dtype)
    cv.mixChannels([hsvt], [hue2], ch)
    hist2 = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    backproj = cv.calcBackProject([hue2], [0], hist, ranges, scale=1)
    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(bins):
        cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)

    cv.imshow('Histogram', histImg) 
    return backproj



def print_center(cnt,target):
    hull = cv.convexHull(cnt)
    drawing=target
    #cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)
    cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
    M = cv.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
    cv.circle(drawing, (cX, cY), 7, (255, 255, 255), -1)
    cv.putText(drawing, "center", (cX - 20, cY - 20),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow('Target',drawing)
    return [cX,cY]

#Function takes in a directory location, searches for image files in it and back projects each one of them to the target, looking for the image with max
def hist_back_prop_max(directory,target):
    
    src=os.listdir(directory)
    if src is None:
        print('Could not find any image in dir.', args.input)
        exit(0)

    k=0


    cv.morphologyEx(target, cv.MORPH_OPEN, kernel)
    mxAr=0
    for file in src:
        j=file.split('.')
        if len(j)!=2 or j[1] not in ['jpg','jpeg','png','tif']:
            continue
        else:
            k=1
        hist_im=cv.imread(args.input+'/'+file)
        cnt=find_max_cnt(find_thresh(hist_im,target,bins))
        ar=cv.contourArea(cnt)
        if ar>mxAr:
            mxAr=ar
            gcnt=cnt
            print(file)

    if k==0:
        print("Sorry no image file exists in the specified module. Exiting...")
        exit(0)
    if mxAr!=0:
        print (print_center(gcnt, target))
    cv.waitKey()



def hist_back_prop_sum(directory,target):
    
    src=os.listdir(directory)
    if src is None:
        print('Could not find any image in dir.', args.input)
        exit(0)

    k=0


    cv.morphologyEx(target, cv.MORPH_OPEN, kernel)
    mxAr=0
    thresh=np.zeros(target.shape[0:2], dtype=np.uint8)
    for file in src:
        j=file.split('.')
        if len(j)!=2 or j[1] not in ['jpg','jpeg','png','tif']:
            continue
        else:
            k=1
        hist_im=cv.imread(args.input+'/'+file)
        thresh=thresh+find_thresh(hist_im,target,bins)


    if k==0:
        print("Sorry no image file exists in the specified module. Exiting...")
        exit(0)
    print (print_center(find_max_cnt(thresh), target))
    cv.waitKey()




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Histo back projection for Humanoid IITK')
    parser.add_argument('--input', help='Path to input module')
    args = parser.parse_args()
    hist_back_prop_sum(args.input,target)