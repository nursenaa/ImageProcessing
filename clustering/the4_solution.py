import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage import filters, segmentation, color,draw
from skimage.future import graph
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.tree import DecisionTreeRegressor









INPUT_PATH = "C:/Users/Ecem Ataman/Desktop/THE4_Images/"
OUTPUT_PATH = "C:/Users/Ecem Ataman/Desktop/THE4_outputs/"

def objectcounting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a black and white image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform morphological opening to remove small noise
    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Perform morphological closing to fill in small holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    counter, _ = cv2.connectedComponents(closing)
# Perform morphological dilation to make the objects more distinct
    dilation = cv2.dilate(closing, kernel, iterations=5)

# Find the contours in the image
    contours,hierarchy= cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    black_contours =cv2.drawContours(black_image, contours, -1, (255, 255, 255), -1)
    counter, _ = cv2.connectedComponents(dilation)
# Draw the contours on the original image
    image_with_contours = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    plt.imshow(image_with_contours)
    plt.show()
    print(f'The number of flowers in image AX is {counter}')
    return  black_contours;

# Save the image with contours
def find_rag(img):
    labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    g = graph.rag_mean_color(img, labels)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

    ax[0].set_title('RAG drawn with default settings')
    lc = graph.show_rag(labels, g, img, ax=ax[0])
    # specify the fraction of the plot area that will be used to draw the colorbar
    fig.colorbar(lc, fraction=0.03, ax=ax[0])

    ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
    lc = graph.show_rag(labels, g, img,
                        img_cmap='gray', edge_cmap='viridis', ax=ax[1])
    fig.colorbar(lc, fraction=0.03, ax=ax[1])

    for a in ax:
        a.axis('off')

    plt.tight_layout()



def tree_relationship_structure(image):
    tree_structure = []
    branch_structure = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                # check if current pixel is a branch or leaf
            if image[i, j].all() == 0:  # branch
                    # initialize empty list to store relationship structure for current branch
                branch_structure = []
                # check if there are any pixels to the left, right, up, or down of the current branch
            if i > 0 and image[i - 1, j].all == 1:  # leaf to the left
                branch_structure.append((i - 1, j))
            if i < image.shape[0] - 1 and image[i + 1, j].all == 1:  # leaf to the right
                branch_structure.append((i + 1, j))
            if j > 0 and image[i, j - 1].all == 1:  # leaf above
                branch_structure.append((i, j - 1))
            if j < image.shape[1] - 1 and image[i, j + 1].all == 1:  # leaf below
                branch_structure.append((i, j + 1))
        # add relationship structure for current branch to tree structure list
    tree_structure.append(branch_structure)



    return tree_structure

def mean_shift_segmentation25(img,sp):
    segmented_img = cv2.pyrMeanShiftFiltering(img, sp=sp, sr=10)
    segmentation_map = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
    # if pixel is not in the same cluster as the original image, color it black
            if not np.array_equal(img[i][j], segmented_img[i][j]):
                segmentation_map[i][j] = [0, 0, 0]
    # otherwise, color it white
            else:
                segmentation_map[i][j] = [255, 255, 255]

    #seg1 = cv2.threshold(mean_shift, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    seg1_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    boundary1 = cv2.Canny(segmented_img, 50, 150)
    #_, tree1, _ = cv2.findContours(seg1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tree1 = tree_relationship_structure(segmented_img)
    #rag1 = cv2.connectedComponents(seg1_gray)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(segmentation_map)
    axs[1].set_title('Segmentation Map (Bandwidth = 20)')
    # Perform mean shift segmentation on the image
    axs[2].imshow(boundary1)
    axs[2].set_title('Boundary Overlay (Bandwidth = 20')
    axs[3].imshow(tree1)
    axs[3].set_title('Tree (Bandwidth = 20')

def mean_shift_segmentation20(img,sp):
        segmented_img = cv2.pyrMeanShiftFiltering(img, sp=sp, sr=10)
        segmentation_map = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # if pixel is not in the same cluster as the original image, color it black
                if not np.array_equal(img[i][j], segmented_img[i][j]):
                    segmentation_map[i][j] = [0, 0, 0]
                # otherwise, color it white
                else:
                    segmentation_map[i][j] = [255, 255, 255]

        # seg1 = cv2.threshold(mean_shift, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        seg1_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        boundary1 = cv2.Canny(segmented_img, 50, 150)
        # _, tree1, _ = cv2.findContours(seg1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tree1 = tree_relationship_structure(segmented_img)
        # rag1 = cv2.connectedComponents(seg1_gray)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img)
        axs[0].set_title('Original Image')
        axs[1].imshow(segmentation_map)
        axs[1].set_title('Segmentation Map (Bandwidth = 20)')
        # Perform mean shift segmentation on the image
        axs[2].imshow(boundary1)
        axs[2].set_title('Boundary Overlay (Bandwidth = 20')
        axs[3].imshow(tree1)
        axs[3].set_title('Tree (Bandwidth = 20')


def mean_shift_segmentation15(img,sp):
    segmented_img = cv2.pyrMeanShiftFiltering(img, sp=sp, sr=10)
    segmentation_map = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
    # if pixel is not in the same cluster as the original image, color it black
            if not np.array_equal(img[i][j], segmented_img[i][j]):
                segmentation_map[i][j] = [0, 0, 0]
    # otherwise, color it white
            else:
                segmentation_map[i][j] = [255, 255, 255]

    #seg1 = cv2.threshold(mean_shift, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    seg1_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    boundary1 = cv2.Canny(segmented_img, 50, 150)
    #_, tree1, _ = cv2.findContours(seg1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tree1 = tree_relationship_structure(segmented_img)
    #rag1 = cv2.connectedComponents(seg1_gray)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(segmentation_map)
    axs[1].set_title('Segmentation Map (Bandwidth = 20)')
    # Perform mean shift segmentation on the image
    axs[2].imshow(boundary1)
    axs[2].set_title('Boundary Overlay (Bandwidth = 20')
    axs[3].imshow(tree1)
    axs[3].set_title('Tree (Bandwidth = 20')

def ncut_segmentation3(img,sigma):
    segments = segmentation.slic(img, 200, compactness=10, sigma=sigma)

    # Create subplot with original image
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[0].set_title("Original Image")

    # Create subplot with segmentation map
    segmentation_map = color.label2rgb(segments, img, kind='avg')
    ax[1].imshow(segmentation_map)
    ax[1].set_title("Segmentation Map")

    # Create subplot with boundary overlay
    boundary_overlay = segmentation.mark_boundaries(img, segments)
    ax[2].imshow(boundary_overlay)
    ax[2].set_title("Boundary Overlay")

    # fit the tree on the segments
    tree_structure = tree_relationship_structure(img)# predict segments for each pixel
    ax[3].imshow(tree_structure)
    ax[3].set_title("Tree Relationship Structure")
    return fig





    # Save the figure as a single .png file

def ncut_segmentation2(img,sigma):
    segments = segmentation.slic(img, 250, compactness=10, sigma=sigma)

    # Create subplot with original image
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[0].set_title("Original Image")

    # Create subplot with segmentation map
    segmentation_map = color.label2rgb(segments, img, kind='avg')
    ax[1].imshow(segmentation_map)
    ax[1].set_title("Segmentation Map")

    # Create subplot with boundary overlay
    boundary_overlay = segmentation.mark_boundaries(img, segments)
    ax[2].imshow(boundary_overlay)
    ax[2].set_title("Boundary Overlay")

    # fit the tree on the segments
    tree_structure = tree_relationship_structure(img)# predict segments for each pixel
    ax[3].imshow(tree_structure)
    ax[3].set_title("Tree Relationship Structure")
    return fig



    # Save the figure as a single .png file

def ncut_segmentation1(img,sigma):
    segments = segmentation.slic(img, 250, compactness=10, sigma=sigma)

    # Create subplot with original image
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[0].set_title("Original Image")

    # Create subplot with segmentation map
    segmentation_map = color.label2rgb(segments, img, kind='avg')
    ax[1].imshow(segmentation_map)
    ax[1].set_title("Segmentation Map")

    # Create subplot with boundary overlay
    boundary_overlay = segmentation.mark_boundaries(img, segments)
    ax[2].imshow(boundary_overlay)
    ax[2].set_title("Boundary Overlay")

    # fit the tree on the segments
    tree_structure = tree_relationship_structure(img)# predict segments for each pixel
    ax[3].imshow(tree_structure)
    ax[3].set_title("Tree Relationship Structure")




    # Save the figure as a single .png file


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    ##objectcounting
    img = cv2.imread(INPUT_PATH + "A1.png")
    img1 = objectcounting(img)
    cv2.imwrite(OUTPUT_PATH + 'A1.png', img1)
    img = cv2.imread(INPUT_PATH + "A2.png")
    img1 = objectcounting(img)
    cv2.imwrite(OUTPUT_PATH + 'A2.png', img1)
    img = cv2.imread(INPUT_PATH + "A3.png")
    img1 = objectcounting(img)
    cv2.imwrite(OUTPUT_PATH + 'A3.png', img1)
    ###MEANSHIFT
    img = cv2.imread(INPUT_PATH + "B1.jpg")

    mean_shift_segmentation25(img,25)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_meanshift_25.png')
    mean_shift_segmentation20(img,20)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_meanshift_20.png')
    mean_shift_segmentation15(img,15)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_meanshift_15.png')
    img = cv2.imread(INPUT_PATH + "B2.jpg")
    mean_shift_segmentation25(img,25)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_meanshift_25.png')
    mean_shift_segmentation20(img,20)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_meanshift_20.png')
    mean_shift_segmentation15(img,15)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_meanshift_15.png')
    img = cv2.imread(INPUT_PATH + "B3.jpg")
    mean_shift_segmentation25(img,25)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_meanshift_25.png')
    mean_shift_segmentation20(img,20)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_meanshift_20.png')
    mean_shift_segmentation15(img,15)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_meanshift_15.png')
    img = cv2.imread(INPUT_PATH + "B4.jpg")
    mean_shift_segmentation25(img,25)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_meanshift_25.png')
    mean_shift_segmentation20(img,20)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_meanshift_20.png')
    mean_shift_segmentation15(img,15)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_meanshift_15.png')
#################NCUTSEGMENTATON
    img = cv2.imread(INPUT_PATH + "B1.jpg")
    find_rag(img)
    plt.savefig(OUTPUT_PATH + 'ragB1')
    ncut_segmentation1(img,1)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_ncut_1.png')

    imgg= ncut_segmentation2(img,2)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_ncut_2.png')
    ncut_segmentation3(img,3)
    plt.savefig(OUTPUT_PATH + 'B1_algorithm_ncut_3.png')
    img = cv2.imread(INPUT_PATH + "B2.jpg")
    find_rag(img)
    plt.savefig(OUTPUT_PATH + 'ragB2')
    imgs=ncut_segmentation1(img,1)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_ncut_1.png')
    imgf=ncut_segmentation2(img,2)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_ncut_2.png')
    imgh=ncut_segmentation3(img,3)
    plt.savefig(OUTPUT_PATH + 'B2_algorithm_ncut_3.png')
    img = cv2.imread(INPUT_PATH + "B3.jpg")
    find_rag(img)
    plt.savefig(OUTPUT_PATH + 'ragB3')
    imgs = ncut_segmentation1(img,1)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_ncut_1.png')
    imgf = ncut_segmentation2(img,2)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_ncut_2.png')
    imgh = ncut_segmentation3(img,3)
    plt.savefig(OUTPUT_PATH + 'B3_algorithm_ncut_3.png')
    img = cv2.imread(INPUT_PATH + "B4.jpg")
    find_rag(img)
    plt.savefig(OUTPUT_PATH + 'ragB4')
    imgs = ncut_segmentation1(img,1)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_ncut_1.png')
    imgf = ncut_segmentation2(img,2)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_ncut_2.png')
    imgh = ncut_segmentation3(img,3)
    plt.savefig(OUTPUT_PATH + 'B4_algorithm_ncut_3.png')



