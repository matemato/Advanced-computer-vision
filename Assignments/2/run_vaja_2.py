import cv2
import time
import numpy as np
import matplotlib . pyplot as plt
from ex2_utils import generate_responses_1, create_epanechnik_kernel

def test():
    fig = plt.figure()
    h = generate_responses_1()
    plt.imshow()
    plt.show()

    fig = plt.figure()
    plt.imshow(generate_responses_1())
    plt.show()

if __name__ == "__main__":
    create_epanechnik_kernel(9,9, 1)
    # test()


