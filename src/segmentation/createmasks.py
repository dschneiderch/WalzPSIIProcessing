from plantcv import plantcv as pcv
import os
import numpy as np
import cv2 as cv2
from skimage import filters

def vismask(img):

    a_img = pcv.rgb2gray_lab(img,channel='a')
    thresh_a = pcv.threshold.binary(a_img, 126, 255, 'dark')

    mask = pcv.fill(thresh_a,1000)
    final_mask = pcv.dilate(mask,2,1)

    return final_mask


def psIImask(img, mode='thresh'):
    # pcv.plot_image(img)
    if mode is 'thresh':

        # this entropy based technique seems to work well when algae is present
        algaethresh = filters.threshold_yen(image=img)
        threshy = pcv.threshold.binary(img, algaethresh, 255, 'light')
        # mask = pcv.dilate(threshy, 2, 1)
        mask = pcv.fill(threshy, 250)
        mask = pcv.erode(mask, 2, 1)
        mask = pcv.fill(mask, 250)
        final_mask = mask  # pcv.fill(mask, 270)

    elif isinstance(mode, pd.DataFrame):
        mode = curvedf
        rownum = mode.imageid.values.argmax()
        imgdf = mode.iloc[[1, rownum]]
        fm = cv2.imread(imgdf.filename[0])
        fmp = cv2.imread(imgdf.filename[1])
        npq = np.float32(np.divide(fm, fmp, where=fmp != 0) - 1)
        npq = np.ma.array(fmp, mask=fmp < 200)
        plt.imshow(npq)
        # pcv.plot_image(npq)

        final_mask = np.zeroes(np.shape(img))

    else:
        pcv.fatal_error('mode must be "thresh" (default) or an object of class pd.DataFrame')

    return final_mask
