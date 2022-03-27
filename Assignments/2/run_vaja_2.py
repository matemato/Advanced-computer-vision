import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from ex2_utils import generate_responses_1, custom_generate_responses_1, get_patch

def mean_shift(pdf, max=[70, 50]):

    plt.imshow(pdf)
    plt.show()
    start = [80, 40]
    real_start = [80, 40]

    size = [15, 15]
    [X, Y] = np.meshgrid(np.arange(-int(size[0]/2), int(size[0]/2) + 1), np.arange(-int(size[1]/2), int(size[1]/2) + 1))

    for i in range(200):
        w,_ = get_patch(pdf, start, size)
        w = w/np.sum(w)
        x_shift = np.sum(np.multiply(w, X))
        y_shift = np.sum(np.multiply(w, Y))
        start[0] += x_shift
        start[1] += + y_shift

        # if np.sqrt(x_shift**2 + y_shift**2) < 0.5:
        #     print(i)
        #     break

    plt.imshow(pdf)
    plt.plot(start[0],start[1], 'rx')
    plt.plot(max[1],max[0], 'gx')
    plt.plot(real_start[0],real_start[1], 'ro')
    plt.show()

def my_mean_shift():
    pass

if __name__ == "__main__":
    mean_shift(generate_responses_1())
    for h in range(10):
        pdf, max = custom_generate_responses_1(random.Random(h))
        mean_shift(pdf, max)


