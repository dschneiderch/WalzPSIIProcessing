from plantcv import plantcv as pcv
import os
import numpy as np
import cv2 as cv2

def vismask(img):

    a_img = pcv.rgb2gray_lab(img,channel='a')
    thresh_a = pcv.threshold.binary(a_img, 126, 255, 'dark')

    mask = pcv.fill(thresh_a,1000)
    final_mask = pcv.dilate(mask,2,1)

    return final_mask


def psIImask(img):

    thresh = pcv.threshold.binary(img, 15, 255, 'light')
    #check thresholded image is now binary whether there is no data or threshold is too high
    if len(np.unique(thresh)) != 2:
        final_mask = np.zeros(np.shape(img)[:2], dtype=np.uint8)
    else:
        mask = pcv.fill(thresh,175)
        final_mask = pcv.erode(mask,2,1)

    return final_mask
