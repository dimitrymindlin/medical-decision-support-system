# Python 2/3 compatibility
import numpy as np
import cv2 as cv
import imutils
import tensorflow as tf

SHOW = False


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img, min_area=100000, max_skew=0.45):
    """
    A method to find inner square images on bigger images
    :param min_area: specifies minimal square area in pixels
    :param max_skew: specifies maximum skewness of squares
    :param img: image tensor
    :return: list of found squares
    """

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(bin)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) >= 4 and cv.contourArea(cnt) >= min_area \
                        and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4],
                                                cnt[(i + 2) % 4])
                                      for i in range(4)])
                    if max_cos < max_skew:
                        squares.append(cnt)
    return squares


def crop_squares(squares, img):
    try:
        rect = cv.minAreaRect(squares[0])
    except IndexError:
        return img
    box = cv.boxPoints(rect)
    box = np.int0(box)

    if SHOW:
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, M, (width, height))

    if width > height:
        warped = imutils.rotate_bound(warped, 270)

    if SHOW:
        cv.imshow("crop_img.jpg", warped)

    return warped

def crop_image(image):
    image = image.numpy().decode("utf-8")
    squares = find_squares(image)
    cv.drawContours(image, squares, 0, (0, 255, 0), 3)
    image = crop_squares(squares, image)
    return tf.cast(tf.convert_to_tensor(image), tf.float32)


