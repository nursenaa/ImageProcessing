import cv2
import os
import matplotlib.pyplot as plt
from skimage import transform

from skimage import img_as_float
import math
import numpy as np

from skimage import data
from skimage import exposure
from scipy.ndimage import interpolation

INPUT_PATH = "C:/Users/SENA NUR/Desktop/the1-images/"
OUTPUT_PATH = "C:/Users/SENA NUR/Desktop/outputs/"

def read_image(img_path, rgb=True):
    img = cv2.imread(img_path)
    return img

def write_image(img, output_path, rgb=True):
    cv2.imwrite(output_path, img)

def extract_save_histogram(img, path):
    plt.subplot(223),
    plt.hist(img.ravel(), bins=50)
    plt.savefig(path)
    plt.close()

def rotate_image(img, degree=0, interpolation_type="linear"):
    h, w = img.shape[:2]

    cX, cY = (w // 2, h // 2) #center x, center y
    if interpolation_type=="cubic":
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def histogram_equalization(img):

    img_hist_eq = cv2.equalizeHist(img)
    return img_hist_eq

def adaptive_histogram_equalization(img):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_adapted = clahe.apply(img)
    return img_adapted;

if _name_ == '_main_':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #PART1

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")

    #PART2
    img = read_image(INPUT_PATH + "b1.png", rgb = False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_image(gray, OUTPUT_PATH + "gray_image1.png")
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)