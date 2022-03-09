import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import matplotlib.pyplot as plt


def image_stack(image_raw, image):
    return np.hstack((image_raw, image))


def show_diff_in_pngs(original, second):
    original_image = original
    counterfactual_image = second

    # convert the images to grayscale
    grayA = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(counterfactual_image, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(counterfactual_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    diff = np.stack((diff,)*3, axis=-1) # to 3 channel
    return diff


def get_diff_maps_for_imgs(real_img, counterfactual_img_list):
    diff_imgs = []
    real_img = real_img / 255
    for fake in counterfactual_img_list:
        diff_imgs.append(show_diff_in_pngs(real_img, fake))
    return diff_imgs
