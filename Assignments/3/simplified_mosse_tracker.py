import cv2
import time
import random
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from ex3_utils import create_cosine_window, create_gauss_peak
from ex2_utils import Tracker, get_patch

class SimplifiedMOSSETracker(Tracker):
    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = [int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)]
        self.patch_size = np.array([int(region[2]), int(region[3])])
        region[2] = int(region[2]*1.2)
        region[3] = int(region[3]*1.2)
        if region[2]%2==0: region[2] -= 1
        if region[3]%2==0: region[3] -= 1
        self.size = np.array([int(region[2]), int(region[3])])
        

        self.win = create_cosine_window(self.size)
        p,_ = get_patch(image, self.position, self.size)
        # plt.imshow(p)
        # plt.show()

        F_hat = fft2(self.win * p)
        # plt.imshow(ifft2(F_hat).astype(float))
        # plt.show()
        self.G_hat = fft2(create_gauss_peak(self.size, self.parameters.sigma))
        # plt.imshow(ifft2(self.G_hat).astype(float))
        # plt.show()
        self.H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.parameters.lamb)
        # plt.imshow(ifft2(self.H_hat).astype(float))
        # plt.show()

        # plt.imshow(ifft2(F_hat * self.H_hat).astype(float))
        # plt.show()


    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p,_ = get_patch(image, self.position, self.size)

        

        F_hat = fft2(self.win * p)

        # plt.imshow(ifft2(F_hat).astype(float))
        # plt.show()

        R = ifft2(self.H_hat * F_hat)

        # plt.imshow(R.astype(float))
        # plt.show()

        # self.position = np.array(np.unravel_index(R.argmax(), R.shape))
        new_position = np.array(np.unravel_index(R.argmax(), R.shape))
        

        if new_position[0] > self.size[1] / 2: new_position[0] -= self.size[1]
        if new_position[1] > self.size[0] / 2: new_position[1] -= self.size[0]

        print(new_position)

        self.position[0] += new_position[1]
        self.position[1] += new_position[0]

        H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.parameters.lamb)

        self.H_hat = (1-self.parameters.alpha) * self.H_hat + self.parameters.alpha * H_hat

        return [self.position[0] - self.patch_size[0]/2, self.position[1] - self.patch_size[1]/2, self.patch_size[0], self.patch_size[1]]

        

class SimplifiedMOSSEParams():
    def __init__(self):
        self.sigma = 1
        self.alpha = 0.01
        self.lamb = 1e-3