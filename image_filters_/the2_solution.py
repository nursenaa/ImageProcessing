import cv2
import os
import matplotlib.pyplot as plt
from skimage import transform
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage import img_as_float
import math
from skimage.filters import difference_of_gaussians, window
import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import dct

from skimage import data
from skimage import exposure
from scipy.ndimage import gaussian_filter
from sympy import fwht
from scipy.linalg import hadamard
from itertools import product
from PIL import Image as im
INPUT_PATH = "C:/Users/SENA NUR/Desktop/THE_2_images/"
OUTPUT_PATH = "C:/Users/SENA NUR/Desktop/the2outputs/"

def read_image(img_path, rgb=True):
    img = cv2.imread(img_path)


    return img

def write_image(img, output_path, rgb=True):
    cv2.imwrite(output_path, img)
def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def fourier(img):
    f = np.fft.fft2(img)
    fshif = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshif))
    return magnitude_spectrum;

    return fshif

def Hadamardm(N):
    if N>2:
        Matrice1=np.concatenate((Hadamardm(N/2),Hadamardm(N/2)),axis=1,out=None)
        Matrice2=np.concatenate((Hadamardm(N/2),-Hadamardm(N/2)),axis=1,out=None)
        Matrice3=np.concatenate((Matrice1,Matrice2),axis=0,out=None)
        H=Matrice3
    else:
        aux2=1
        A=[[1,1],[1,-1]]
        H=np.dot(aux2,A)

    return H
def hadamard(img):
    size=(1024,1024);
    N=1024;

    img2 = cv2.resize(img, size)
    HH=Hadamardm(N);
    var = 1 / math.sqrt(N)
    HH = np.dot(var, HH)


    aux = np.dot(HH, np.asarray(img2))  # H*U
    V = np.dot(aux,HH)

    V =im.fromarray(V);# (H*U)*H
    return V



def cosine(img):
    imf = np.float32(img) / 255.0  # float conversion/scale
    dct = cv2.dct(imf,cv2.DCT_INVERSE)  # the dct
    dctImage = np.uint8(dct * 255.0)
    return dctImage;



def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
def ideallowPassFiltering(img,d0):#Transfer parameters are Fourier transform spectrogram and filter size
    #h, w = img.shape[0:2]#Getting image properties
    #h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    #img2 = np.zeros((h, w), np.uint8)#Define a blank black image with the same size as the Fourier Transform Transfer
    #img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
    #img3=img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
    #return img3;
    #[M, N] = img.shape[0:2];
    #FT_img = fft2((img));
    #u = np.array([0..M-1]);
    #idx = np.argwhere(u > M / 2);
    #u[idx] = u[idx] - M;
    #v = np.array([0..N-1]);
    #idy = np.argwhere(v > N / 2);
    #v[idy] = v[idy] - N;
    #[V, U] = np.meshgrid(v, u);

    #D = math.sqrt(U^ 2 + V^ 2);
    #H = math.double(D <= d0);
    #G = H* FT_img;
    #output_image = math.real(ifft2(G))

    base = np.zeros(img.shape[:2])
    rows, cols = img.shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < d0:
                base[y, x] = 1
    return base;


def gen_gaussian_kernel_low(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]

    g = 1 / (2 * np.pi * sigma**2) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return g

def gaussianlow(img,k_size,D0):
    height, width = img.shape[0], img.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = np.ravel(img[i: i + k_size, j: j + k_size])
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel_low(k_size,D0)
    filter_array = np.ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst
def butterlowpass(img,D0,n):
    M_shape, N_shape = img.shape[0:2]
    H = np.zeros((M_shape, N_shape))
    # It is order for butterworth
    for u in range(M_shape):
        for v in range(N_shape):
            D = np.sqrt((u - M_shape / 2) ** 2 + (v - N_shape / 2) ** 2)
            H[u, v] = 1 / (1 + (D / D0) ** n)

    return H;
                #H[u][v] = 1 / (1 + x)


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 - np.exp((pow(-x, 2) / (2 * pow(sigma, 2)) - pow(y, 2) / (2 * pow(sigma, 2))));
    return (g)
def gausshigh(img,k_size,sigma):
    height, width = img.shape[0], img.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size pixels into a row and np.vstack all rows
    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i, j in product(range(dst_height), range(dst_width)):
        window = np.ravel(img[i: i + k_size, j: j + k_size])
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel =gen_gaussian_kernel(k_size, sigma)
    filter_array = np.ravel(gaussian_kernel)

    # reshape and get the dst image
    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst

def highPassFiltering(img,d0):#Transfer parameters are Fourier transform spectrogram and filter size
    base = np.ones(img.shape[:2])
    rows, cols = img.shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < d0:
                base[y, x] = 0
    return base
def butterhigh(img,D0,n):

    M_shape, N_shape = img.shape[0:2]
    H = np.zeros((M_shape, N_shape))
      # It is order for butterworth
    for u in range(M_shape):
        for v in range(N_shape):
            D = np.sqrt((u - M_shape / 2) ** 2 + (v - N_shape / 2) ** 2)
            H[u, v] = 1 / (1 + (D / D0) ** n)

    return (1-H);


def band_reject_filter(img):
    P, Q = img.shape[0:2];
    print(P);
    print(Q);

    H = np.ones((P, Q))


    for u in range(P):
       for v in range(Q):
            # Get euclidean distance from point D(u,v) to the center
          dis1 = distance((u, v), (P / 2, Q / 2));

          if (dis1 >50 and dis1 < 100) or (dis1 >200  and dis1 < 300):
               H[u][v] = 0


    return H

def band_pass_filter(img):
    P, Q = img.shape[0:2]
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center

            dis1 = distance((u, v), (P / 2, Q / 2));


            if (dis1 >=50 and dis1 <= 90) or (dis1 > 200 and dis1 < 400):
                H[u][v] = 1
            else:
                H[u][v] = 0

    return H

def averaging(img):
    img=cv2.blur(img, (5,5))
    im2 = cv2.boxFilter(img, -1, (2, 2), normalize=True)
    return im2;

def gaussiann(img):
    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    return dst;
def median(img):
    dst= cv2.medianBlur(img, 7)
    return dst;
def normalize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgg = clahe.apply(img)
    return imgg






    # Loop over the image and apply Min-Max formulae

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #ımage1

    img = read_image(INPUT_PATH + "image1.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    foutput=fourier(blue)
    goutput=fourier(green)
    routput=fourier(red)


    foutput = np.dstack((np.abs(foutput), np.abs(goutput), np.abs(routput)))


    write_image(foutput, OUTPUT_PATH + "F1.png")

    bhoutput = hadamard(blue)
    rhoutput=hadamard(red)
    ghoutput=hadamard(green)
    output = np.dstack((np.abs(bhoutput), np.abs(rhoutput), np.abs(ghoutput)))

    write_image(output, OUTPUT_PATH +"H1.png")
    boutput=cosine(blue)
    routput=cosine(red)
    goutput=cosine(green)
    coutput = np.dstack((np.abs(boutput), np.abs(routput), np.abs(goutput)))

    write_image(coutput, OUTPUT_PATH + "C1.png")
   #image2
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    foutput = fourier(blue)
    goutput = fourier(green)
    routput = fourier(red)
    img = read_image(INPUT_PATH + "2.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    boutput = fourier(blue)
    goutput = fourier(green)
    routput = fourier(red)

    foutput = np.dstack((np.abs(boutput), np.abs(routput), np.abs(goutput)))
    write_image(foutput, OUTPUT_PATH + "F2.png")

    bhoutput = hadamard(blue);
    ghoutput = hadamard(green);
    rhoutput = hadamard(red);
    hhoutput = np.dstack((np.abs(bhoutput), np.abs(ghoutput), np.abs(rhoutput)))
    write_image(hhoutput, OUTPUT_PATH + "H2.png")
    bcoutput =cosine(blue);
    gchoutput = cosine(green);
    rchoutput = cosine(red);
    ccoutput = np.dstack((np.abs(bcoutput), np.abs(gchoutput), np.abs(rchoutput)))

    write_image(ccoutput, OUTPUT_PATH + "C2.png")
 #image3

    img = read_image(INPUT_PATH + "3.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = ideallowPassFiltering(blue,30)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = ideallowPassFiltering(green, 30)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = ideallowPassFiltering(red, 30)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(ilppoutput, OUTPUT_PATH + "ILP_r1.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = ideallowPassFiltering(blue, 60)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = ideallowPassFiltering(green, 60)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = ideallowPassFiltering(red, 60)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(ilppoutput, OUTPUT_PATH + "ILP_r2.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = ideallowPassFiltering(blue, 90)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = ideallowPassFiltering(green, 90)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = ideallowPassFiltering(red, 90)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))


    write_image(ilppoutput, OUTPUT_PATH + "ILP_r3.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gaussbloutput = gaussianlow(blue, 4, 6)
    gaussgloutput = gaussianlow(green, 4, 6)
    gaussrloutput = gaussianlow(red, 4, 6)
    sgilppoutput = np.dstack((np.abs(gaussbloutput), np.abs(gaussgloutput), np.abs(gaussrloutput)))

    write_image(sgilppoutput, OUTPUT_PATH + "GLP_r1.png")

    gaussbloutput = gaussianlow(blue, 8, 6)
    gaussgloutput = gaussianlow(green, 8, 6)
    gaussrloutput = gaussianlow(red, 8, 6)
    gilppoutput = np.dstack((np.abs(gaussbloutput), np.abs(gaussgloutput), np.abs(gaussrloutput)))
    write_image(gilppoutput, OUTPUT_PATH + "GLP_r2.png")

    ggaussbloutput = gaussianlow(blue, 12, 6)
    ggaussgloutput = gaussianlow(green, 12, 6)
    ggaussrloutput = gaussianlow(red, 12, 6)
    gilppoutput = np.dstack((np.abs(ggaussbloutput), np.abs(ggaussgloutput), np.abs(ggaussrloutput)))
    write_image(gilppoutput, OUTPUT_PATH + "GLP_r3.png")
    img = read_image(INPUT_PATH + "3.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterlowpass(blue, 30,2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterlowpass(green, 30,2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = butterlowpass(red, 30,2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(ilppoutput, OUTPUT_PATH + "BLP_r1.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterlowpass(blue, 60,2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterlowpass(green, 60,2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput =butterlowpass(red, 60,2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))

    write_image(ilppoutput, OUTPUT_PATH + "BLP_r2.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterlowpass(blue, 90, 2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterlowpass(green, 90, 2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = butterlowpass(red, 90, 2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    ilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(ilppoutput, OUTPUT_PATH + "BLP_r3.png")

    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = highPassFiltering(blue, 30)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = highPassFiltering(green,30)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = highPassFiltering(red, 30)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(bilppoutput, OUTPUT_PATH + "IHP_r1.png")

    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = highPassFiltering(blue, 60)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = highPassFiltering(green, 60)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = highPassFiltering(red, 60)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(bilppoutput, OUTPUT_PATH + "IHP_r2.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = highPassFiltering(blue, 90)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = highPassFiltering(green, 90)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = highPassFiltering(red, 90)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))

    write_image(bilppoutput, OUTPUT_PATH + "IHP_r3.png")
    img = read_image(INPUT_PATH + "3.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gaussbloutput = gausshigh(blue, 6, 10)
    gaussgloutput = gausshigh(green, 6, 10)
    gaussrloutput = gausshigh(red, 6, 10)
    bgilppoutput = np.dstack((np.abs(gaussbloutput), np.abs(gaussgloutput), np.abs(gaussrloutput)))
    write_image(bgilppoutput, OUTPUT_PATH + "GHP_r1.png")
    ggaussbloutput = gausshigh(blue, 8, 10)
    ggaussgloutput = gausshigh(green, 8, 10)
    ggaussrloutput = gausshigh(red, 8, 10)
    ggilppoutput = np.dstack((np.abs(ggaussbloutput), np.abs(ggaussgloutput), np.abs(ggaussrloutput)))
    write_image(ggilppoutput, OUTPUT_PATH + "GHP_r2.png")
    rgaussbloutput = gausshigh(blue, 10, 10)
    rgaussgloutput = gausshigh(green, 10, 10)
    rgaussrloutput = gausshigh(red, 10, 10)
    rgilppoutput = np.dstack((np.abs(rgaussbloutput), np.abs(rgaussgloutput), np.abs(rgaussrloutput)))
    write_image(rgilppoutput, OUTPUT_PATH + "GHP_r3.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterhigh(blue, 30, 2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterhigh(green, 30, 2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = butterhigh(red, 30,2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))

    write_image(bilppoutput, OUTPUT_PATH + "BHP_r1.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterhigh(blue, 60,2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterhigh(green, 60,2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = butterhigh(red, 60,2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(bilppoutput, OUTPUT_PATH + "BHP_r2.png")
    npImg = np.array(blue)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealboutput = butterhigh(blue, 90,2)
    idealboutput = idealboutput * imgTrans
    idealboutput = np.fft.ifftshift(idealboutput)
    idealboutput = np.fft.ifft2(idealboutput)
    idealboutput = np.abs(idealboutput)
    npImg = np.array(green)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealgoutput = butterhigh(green, 90,2)
    idealgoutput = idealgoutput * imgTrans
    idealgoutput = np.fft.ifftshift(idealgoutput)
    idealgoutput = np.fft.ifft2(idealgoutput)
    idealgoutput = np.abs(idealgoutput)
    npImg = np.array(red)
    imgTrans = np.fft.fftshift(np.fft.fft2(npImg))
    idealroutput = butterhigh(red, 90,2)
    idealroutput = idealroutput * imgTrans
    idealroutput = np.fft.ifftshift(idealroutput)
    idealroutput = np.fft.ifft2(idealroutput)
    idealroutput = np.abs(idealroutput)
    bilppoutput = np.dstack((np.abs(idealboutput), np.abs(idealgoutput), np.abs(idealroutput)))
    write_image(bilppoutput, OUTPUT_PATH + "BHP_r3.png")

    ##image4##
    img = read_image(INPUT_PATH + "4.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    bfour=fourier(blue)
    gfour=fourier(green)
    rfour=fourier(red)
    foout = np.dstack((np.abs(bfour), np.abs(gfour), np.abs(rfour)))
    write_image(foout, OUTPUT_PATH + "BR1_r1.fourier.png")

    img = read_image(INPUT_PATH + "4.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    bbandrej = band_reject_filter(blue);
   # imgTrans = np.fft.fftshift(np.fft.fft2(blue))
   #write_image(imgTrans, OUTPUT_PATH + "BR1_r1trans.png");
    #bbandrej= imgTrans * bbandrej;
    #bbandrej = np.fft.ifftshift(bbandrej)
    #bbandrej = np.fft.ifft2(bbandrej)

    gbandrej = band_reject_filter(green)
    #imgTrans1 = np.fft.fftshift(np.fft.fft2(green))
    #gbandrej = imgTrans1 * gbandrej
    #gbandrej = np.fft.ifftshift(gbandrej)
    #gbandrej = np.fft.ifft2(gbandrej)


    rbandrej = band_reject_filter(red)


    #imgTrans2 = np.fft.fftshift(np.fft.fft2(red))
    #rbandrej = imgTrans2 * rbandrej
    #rbandrej = np.fft.ifftshift(rbandrej)
   # rbandrej = np.fft.ifft2(rbandrej)

    bilppoutput = np.dstack((np.abs(bbandrej), np.abs(gbandrej), np.abs(rbandrej)))
    write_image(bilppoutput, OUTPUT_PATH + "BR1_r1.png")




 ##image5##

    img = read_image(INPUT_PATH + "5.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rbandrej = band_pass_filter(blue);
    imgTrans = np.fft.fftshift(np.fft.fft2(blue))
    bandrej = imgTrans * rbandrej;
    rbandrej = np.fft.ifftshift(bandrej)
    bbandrej = np.fft.ifft2(rbandrej)

    gbandrej = band_pass_filter(green)
    imgTrans1 = np.fft.fftshift(np.fft.fft2(green))
    gbandrej = imgTrans1 * gbandrej
    gbandrej = np.fft.ifftshift(gbandrej)
    gbandrej = np.fft.ifft2(gbandrej)


    rbandrej = band_pass_filter(red)
    imgTrans2 = np.fft.fftshift(np.fft.fft2(red))
    rbandrej = imgTrans2 * rbandrej
    rbandrej = np.fft.ifftshift(rbandrej)
    rbandrej = np.fft.ifft2(rbandrej)

    bilppoutput = np.dstack((np.abs(bbandrej), np.abs(gbandrej), np.abs(rbandrej)))
    write_image(bilppoutput, OUTPUT_PATH + "BP1_r1.png")

#image6##
    img = read_image(INPUT_PATH + "6.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    boutt=median(blue);
    bnormm=normalize(boutt);
    goutt = median(green);
    gnormm = normalize(goutt);
    routt = median(red);
    rnormm = normalize(routt);

    bilspace = np.dstack((np.abs(bnormm), np.abs(gnormm), np.abs(rnormm)))
    write_image(bilspace, OUTPUT_PATH + "Space6.png")
##image7##
    img = read_image(INPUT_PATH + "7.png")
    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    boutt = median(blue);
    bnormm = normalize(boutt);
    goutt = median(green);
    gnormm = normalize(goutt);
    routt = median(red);
    rnormm = normalize(routt);
    bilsspace = np.dstack((np.abs(bnormm), np.abs(gnormm), np.abs(rnormm)))
    write_image(bilsspace, OUTPUT_PATH+"Space7.png")