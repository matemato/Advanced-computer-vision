import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch, extract_histogram

def test():
    im = cv2.imread("./Sequences/ball/00000000.jpg")
    start = [222,134]
    size = [45, 45]
    [X, Y] = np.meshgrid(np.arange(-int(size[0]/2), int(size[0]/2) + 1), np.arange(-int(size[1]/2), int(size[1]/2) + 1))
    wi,_ = get_patch(im, start, size)
    
    h = extract_histogram(wi, 16)
    plt.hist(h)
    plt.show()

    plt.imshow(wi)
    plt.show()



if __name__ == "__main__":
    test()


