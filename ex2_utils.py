from collections import defaultdict
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv, linalg
import scipy as sp
from scipy import ndimage

""" 
    convolve a 1-D array with the given kernel
"""


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    flip_kernel = np.flip(kernel1)  # flip the kernel
    len_ = kernel1.size - 1  # size to pad the kernel
    sign_pad = np.pad(inSignal, (len_, len_), 'constant')  # pad the sides
    conv_sign = np.zeros(inSignal.size + len_)
    for i in range(len_ + inSignal.size):
        conv_sign[i] = np.dot(sign_pad[i:i + len_ + 1], flip_kernel)
    return conv_sign


""" 
    convolve a 2-D array with given kernel 
"""


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    flip_kernel = np.flip(kernel2)  # flip the kernel
    flip_kernel = flip_kernel / np.sum(flip_kernel)
    conv_img = np.zeros(inImage.shape)  # the same size as the original
    len_x, len_y = kernel2.shape[0] // 2, kernel2.shape[1] // 2
    pad_img = cv2.copyMakeBorder(inImage, len_x, len_x, len_y, len_y, cv2.BORDER_REPLICATE)
    # top, bottom, left, right
    for i in range(conv_img.shape[0]):  # run on all the rows
        for j in range(conv_img.shape[1]):  # run on all the columns
            partial_pad = pad_img[i: i + flip_kernel.shape[0], j: j + flip_kernel.shape[1]]
            conv_img[i, j] = np.sum(np.multiply(partial_pad, flip_kernel))
    return conv_img


""" 
    calculate gradient of an image
    @:return direction matrix, magnitude matrix, x_derivative, y_derivative
"""


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    mean_kernel = np.ones(shape=(5, 5))  # mean filter didn't bring good result
    noise_del = cv2.GaussianBlur(inImage, (5, 5), 1)
    kernel_x = np.array([1, 0, -1]).reshape(1, 3)   # create the kernel for x derivative
    kernel_y = np.array([1, 0, -1]).reshape(3, 1)   # create the kernel for y derivative
    x_derivative = cv2.filter2D(noise_del, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)  # derive according to x
    y_derivative = cv2.filter2D(noise_del, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)  # derive according to y
    mag_mat = np.sqrt(np.square(y_derivative) + np.square(x_derivative)).astype('uint8')  # create the magnitude
    direction_mat = np.arctan2(y_derivative, x_derivative)   # the direction of the y/x
    return direction_mat, mag_mat, x_derivative, y_derivative


"""
    Blur an image using a Gaussian kernel
"""


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    if kernel_size[0] % 2 == 1 or kernel_size[1] % 2 == 1:  # check if the kernel is odd
        raise Exception("The kernel size must bee odd number!!")
    kernel = np.zeros(kernel_size)  # define an array
    sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            kernel[i][j] = (1 / 2 * np.pi * np.square(sigma)) * (
                        np.e ** (-1 * ((i ** 2 + j ** 2) / 2 * np.square(sigma))))
    return conv2D(in_image, kernel)


"""
    Blur an image using a Gaussian kernel using OpenCV built-in functions
"""


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    kernel = cv2.getGaussianKernel(kernel_size[0], 0)
    blur = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blur


""" 
    Detects edges using the Sobel method

"""


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    Sx_kernel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])  # multiple of two vectors [1 2 1].T * [1 0 -1]
    Sy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    sobelX = cv2.filter2D(img, -1, Sx_kernel, borderType=cv2.BORDER_REPLICATE)  # convolve the x direction
    sobelY = cv2.filter2D(img, -1, Sy_kernel, borderType=cv2.BORDER_REPLICATE)  # convolve the y direction
    magnitude = np.sqrt(np.square(sobelX) + np.square(sobelY))
    res = np.zeros(img.shape)
    res[magnitude >= thresh] = 1  # check the edge that greater than the threshold
    return sobel(img, thresh), res

"""
Sobel filter using Open CV 
"""
def sobel(img: np.ndarray, thresh: float) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y
    magnitude = cv2.magnitude(sobelx, sobely)
    ans = np.zeros(img.shape)
    ans[magnitude >= thresh] = 1   # check the edge that greater than the threshold
    return ans

"""
Filter That found edges using second derivative
then check zero crossing 
"""
def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    smooth = cv2.GaussianBlur(img, (5, 5), 1)
    kernel = np.array([[0, 1, 0],  # second derivative
                       [1, -4, 1],
                       [0, 1, 0]])
    derivative = cv2.filter2D(smooth, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    res = np.ones(img.shape)
    res = zeroCrossing(derivative, res)
    return res

"""
@:param img 
@:param res - nd.array that use for the picture where that 
ffound {+,0,-} or {-,+} 
"""
def zeroCrossing(img: np.ndarray, res: np.ndarray) -> (np.ndarray, np.ndarray):
    w, h = img.shape
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            window = img[x - 1:x + 2, y - 1:y + 2]  # create (3,3) matrix
            point = img[x, y]
            wmax = window.max()
            wmin = window.min()
            if point == 0.0 and wmax > 0.0 and wmin < 0.0:  # {+,0,-} in every shape
                zeroCross = 0
            elif point > 0.0:  # {+,-} in every shape
                zeroCross = 0 if wmin < 0.0 else 1
            else:
                zeroCross = 0 if wmin < 0.0 else 1
            res[x, y] = zeroCross
    res = res * 255
    return res


""" 
    filter for detecting edges using zero crossing 
"""


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    smooth = cv2.GaussianBlur(img, (5, 5), 1)
    # convolve the smoothed image with the Laplacian filter
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    lap_img = cv2.filter2D(smooth, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    res = np.zeros(img.shape)
    res = zeroCrossing(lap_img, res)  # a binary image (0,1) that representing the edges
    return res


""" 
    Detecting edges usint "Canny Edge" method
"""


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y
    magnitude = cv2.magnitude(sobelx, sobely)
    directions = np.arctan(sobelx, sobely)
    quant_directions = quantGradientDirections(directions)  # find the direct
    del_mul = nonMaxSupression(magnitude, quant_directions)
    res = threshold(del_mul, thrs_1, thrs_2)
    img = (img*255).astype('uint8')
    cv_sol = cv2.Canny(img, thrs_1*255, thrs_2*255)
    return cv_sol, res


"""
quant the values of the directions to four values
"""


def quantGradientDirections(dir: np.ndarray) -> np.ndarray:
    dir = dir % 180  # first need to equalize to opposite directions ---> = <----  -> modulo 180
    dir[np.logical_and(0 <= dir, dir < 22.5)] = 0
    dir[np.logical_and(157 < dir, dir <= 180)] = 0
    dir[np.logical_and(22.5 <= dir, dir < 67.5)] = 45
    dir[np.logical_and(67.5 <= dir, dir < 112.5)] = 90
    dir[np.logical_and(112.5 <= dir, dir < 157.5)] = 135
    return dir


"""
@:param mag - magnitude
@ param dir - direction
"""


def nonMaxSupression(mag: np.ndarray, dir: np.ndarray) -> np.ndarray:
    res = np.zeros(mag.shape)
    x, y = mag.shape
    side1 = 0
    side2 = 0
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            # 45 angle
            if dir[i, j] == 45:
                side1 = mag[i + 1, j - 1]
                side2 = mag[i - 1, j + 1]
            # 90 angle
            if dir[i, j] == 90:
                side1 = mag[i + 1, j]
                side2 = mag[i - 1, j]
            # 135 angle
            if dir[i, j] == 135:
                side1 = mag[i - 1, j - 1]
                side2 = mag[i + 1, j + 1]
            # 180 angle
            if dir[i, j] == 0:
                side1 = mag[i, j + 1]
                side2 = mag[i, j - 1]
            if (mag[i, j] >= side2) and (mag[i, j] >= side1):
                res[i, j] = mag[i, j]
            else:
                res[i, j] = 0
    return res

"""
sign all the pixels that greater than the high threshold as 1 
"""
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    res = img.copy()
    img[img > highThresholdRatio] = 1   # check if the pixel greater than the threshold
    res[img == 1] = 1
    res = hysteresis(img, res, highThresholdRatio, lowThresholdRatio)
    return res

"""
The function care on all the pixels that between the low threshold to high threshold 
"""
def hysteresis(img, res, highThresholdRatio=255, lowThresholdRatio=0.05):
    w, h = img.shape
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            new_point = 0
            window = img[x - 1:x + 2, y - 1:y + 2]  # create (3,3) matrix
            point = img[x, y]
            wmax = window.max()
            if point >= lowThresholdRatio and wmax > highThresholdRatio:
                new_point = 1
            res[x, y] = new_point
    return res


""" 
    Find Circles in an image using a Hough Transform algorithm extension
"""


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    blur_img = cv2.GaussianBlur(img, (5, 5), 1)
    edge_image = cv2.Canny((blur_img*255).astype('uint8'), 50, 150)  # get an image with edges - binary image
    edge_image = edge_image/255
    cv2.imshow('canny', edge_image)
    cv2.waitKey(0)
    # Radius ranges from r_min to r_max
    cols = img.shape[1]
    rows = img.shape[0]
    circles = list()
    for r in range(min_radius, max_radius):  # check all the circles with specific radius
        img_line = np.zeros(img.shape)  # initializing an empty 2D array with zeroes
        for x in range(rows):  # iterating through the original image
            for y in range(cols):
                if edge_image[x][y] == 1:  # if there is edge
                    for angle in range(0, 360):
                        b = y - round(r * np.sin(np.deg2rad(angle)))
                        a = x - round(r * np.cos(np.deg2rad(angle)))
                        if a in range(a, rows) and b in range(0, cols):
                            img_line[a][b] += 1
        maximum = np.amax(img_line)
        if maximum > 150:  # it's show that the image contain circle for this radius
            bin_range(img_line, rows, cols, r, circles)
    return circles


"""
The function check 3*3 range in the image and check their average 
if their average bigger than the threshold it's show that there
have center of circle, otherwise don't sign this point as center of 
circle 
"""


def bin_range(img_line: np.ndarray, rows: np.ndarray, cols: np.ndarray, r: int, circles: list):
    # Initial threshold
    img_line[img_line < 150] = 0
    # find the circles for this radius
    for i in range(1, rows - 1):  # bin - using as range
        for j in range(1, cols - 1):
            if img_line[i][j] >= 150:
                window = img_line[i - 1:i + 2, j - 1: j + 2]
                avg_sum = window.sum() / 9  # bin for range of point
                if avg_sum >= 30:
                    circles.append((i, j, r))
                    img_line[img_line == window] = 0
