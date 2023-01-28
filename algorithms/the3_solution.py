import numpy as np

import cv2
import os

import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2hsv


# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)

INPUT_PATH = "C:/Users/nur20/OneDrive/Desktop/THE3_Images/"
OUTPUT_PATH = "C:/Users/nur20/OneDrive/Desktop/THE3_outputs/"

def read_image(img_path, rgb=True):
    img = cv2.imread(img_path)


    return img

def write_image(img, output_path, rgb=True):
    cv2.imwrite(output_path, img)

# Load the cascade file for face detection


 #   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of skin color in HSV values

  #  lower_skin = np.array([7, 20, 80], dtype=np.uint8)
  #  upper_skin = np.array([255, 255, 255], dtype=np.uint8)

# Create a mask that only selects pixels in the skin color range
   # skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Apply a morphological operation to clean up the mask and remove any small
# isolated regions
   # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# Use the mask to select only the skin-colored pixels in the image
   # skin_pixels = cv2.bitwise_and(img, img, mask=skin_mask)

# Detect faces in the image using a Haar cascade classifier
   # classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   # faces = classifier.detectMultiScale(skin_pixels, 1.1, 5)

# Draw a rectangle around each detected face
    #for (x,y,w,h) in faces:
     #   cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    #return img
##QUESTION I
#def detect1_faces(img):
 #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #  face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
   # faces = face_cascade.detectMultiScale(gray, 1.3,)
   # for (x, y, w, h) in faces:
    #    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
   # return img;

def detect_faces(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([5, 40, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img;

    # ##Q2
def color_images(img,source):
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    color_mapg = np.zeros((256, 3))
    colored_images = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            color_mapg[gray[i, j]] = source[i, j]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            colored_images[i, j] = color_mapg[img[i, j]]

    return colored_images






def plot(img):
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsi[:, :, 0]
    s = hsi[:, :, 1]
    i = hsi[:, :, 2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(h, cmap='gray')
    plt.title('H channel')
    plt.subplot(3, 1, 2)
    plt.imshow(s, cmap='gray')
    plt.title('S channel')
    plt.subplot(3, 1, 3)
    plt.imshow(i, cmap='gray')
    plt.title('I channel')
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(r, cmap='gray')
    plt.title('R channel')
    plt.subplot(3, 1, 2)
    plt.imshow(g, cmap='gray')
    plt.title('G channel')
    plt.subplot(3, 1, 3)
    plt.imshow(b, cmap='gray')
    plt.title('B channel')

    plt.show()
def detect_edges(img, color_space):
    if color_space == "RGB":
        img_color = img
    elif color_space == "HSI":
        img_color = rgb2hsv(img);

    # Calculate gradients in x and y directions
    grad_x = cv2.Sobel(img_color, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_color, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude of gradients
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize gradients
    grad_mag = (grad_mag / grad_mag.max()) * 255

    # Convert gradients to uint8
    grad_mag = grad_mag.astype(np.uint8)

    # Return edge-detected image
    return grad_mag


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    ##face detect
    source = cv2.imread(INPUT_PATH + "1_source.png")
    source=(detect_faces(source))
    cv2.imwrite(OUTPUT_PATH + "1_faces.png", source)

    source = cv2.imread(INPUT_PATH + "2_source.png")
    source = (detect_faces(source))
    cv2.imwrite(OUTPUT_PATH + "2_faces.png", source)
    source = cv2.imread(INPUT_PATH + "3_source.png")
    source = (detect_faces(source))
    cv2.imwrite(OUTPUT_PATH + "3_faces.png", source)

    #pseudocolor

    img = cv2.imread(INPUT_PATH + "1.png",cv2.IMREAD_GRAYSCALE);
    source=cv2.imread(INPUT_PATH + "1_source.png")
    outimg2=color_images(img,source)
    outimg2 = cv2.normalize(outimg2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.cvtColor(outimg2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(OUTPUT_PATH + "1_colored.png", outimg2)
    plot(outimg2)
    img = cv2.imread(INPUT_PATH + "2.png", cv2.IMREAD_GRAYSCALE);
    source = cv2.imread(INPUT_PATH + "2_source.png")
    outimg2 = color_images(img, source)
    outimg2 = cv2.normalize(outimg2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.cvtColor(outimg2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(OUTPUT_PATH + "2_colored.png", outimg2)
    plot(outimg2)
    img = cv2.imread(INPUT_PATH + "3.png", cv2.IMREAD_GRAYSCALE);
    source = cv2.imread(INPUT_PATH + "3_source.png")
    outimg2 = color_images(img, source)
    outimg2 = cv2.normalize(outimg2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.cvtColor(outimg2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(OUTPUT_PATH + "3_colored.png", outimg2)
    plot(outimg2)
    img = cv2.imread(INPUT_PATH + "4.png", cv2.IMREAD_GRAYSCALE);
    source = cv2.imread(INPUT_PATH + "4_source.png")
    outimg2 = color_images(img, source)
    outimg2 = cv2.normalize(outimg2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.cvtColor(outimg2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(OUTPUT_PATH + "4_colored.png", outimg2)
    plot(outimg2)

  ##edge detects

    img = cv2.imread(INPUT_PATH + "3_source.png");
    outimg1 = detect_edges(img,"HSI");
    routimg = detect_edges(img, "RGB");
    cv2.imwrite(OUTPUT_PATH + "3_hcolored_edges.png", outimg1)
    cv2.imwrite(OUTPUT_PATH + "3_rgbcolored_edges.png", routimg)


    img = cv2.imread(INPUT_PATH + "2_source.png")
    outimg = detect_edges(img,"HSI");
    routimg = detect_edges(img, "RGB");
    cv2.imwrite(OUTPUT_PATH + "2_hcolored_edges.png", outimg)
    cv2.imwrite(OUTPUT_PATH + "2_rgbcolored_edges.png", routimg)

    img = cv2.imread(INPUT_PATH + "1_source.png");
    outimg = detect_edges(img,"HSI");
    routimg=detect_edges(img,"RGB");
    cv2.imwrite(OUTPUT_PATH + "1_hcolored_edges.png", outimg)
    cv2.imwrite(OUTPUT_PATH + "1_rgbcolored_edges.png", routimg)




