import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

def zscore(img):
    """Normalization to zero mean and unit variance for removing intensity information."""
    m = np.mean(img)
    s = np.std(img)
    img_norm = (img-m)/s
    
    return img_norm

def contour(img, label):
    """Calculate the contour of the vessels by subtracting the erosion of the label from the label itself.
    Insert the original intensity values of the image in the contour region.
    """

    erosion = ndimage.binary_erosion(label)*1
    subtraction_result = label - erosion

    img_result = np.zeros_like(img)
    img_result[subtraction_result > 0] = img[subtraction_result > 0]
    
    return img_result

def skeletization(img, label):
    """Calculate the skeleton of the vessels by applying the skeletonization algorithm to the label.
    Insert the original intensity values of the image in the skeleton region.
    """

    skeleton = skeletonize(label)
    
    img_result = np.zeros_like(img)
    img_result[skeleton > 0] = img[skeleton > 0]

    return img_result
