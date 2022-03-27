import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch, extract_histogram

def mean_shift():
    pdf = generate_responses_1()
    start = [80, 40]
    real_start = [80, 40]

    size = [31, 31]
    [X, Y] = np.meshgrid(np.arange(-int(size[0]/2), int(size[0]/2) + 1), np.arange(-int(size[1]/2), int(size[1]/2) + 1))

    filtered = generate_responses_1()
    for i in range(20):
        w,_ = get_patch(filtered, start, size)
        w = w/np.sum(w)
        x_shift = np.sum(np.multiply(w, X))
        y_shift = np.sum(np.multiply(w, Y))
        start[0] += x_shift
        start[1] += + y_shift

        if np.sqrt(x_shift**2 + y_shift**2) < 0.5:
            print(i)
            break

    plt.imshow(filtered)
    plt.plot(start[0],start[1], 'rx')
    plt.plot(real_start[0],real_start[1], 'ro')
    plt.show() 

if __name__ == "__main__":
    mean_shift()


