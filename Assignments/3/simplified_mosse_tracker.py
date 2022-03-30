import cv2
import time
import random
import sys
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from ex3_utils import create_cosine_window, create_gauss_peak
from ex2_utils import get_patch#, Tracker
from utils.tracker import Tracker

class SimplifiedMOSSETracker(Tracker):
    def name(self):
        return 'Simplified_MOSSE_Tracker'

    def initialize(self, image, region):

        self.set_params()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = [int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)]
        self.patch_size = np.array([int(region[2]), int(region[3])])
        region[2] = int(region[2]*self.enlarge)
        region[3] = int(region[3]*self.enlarge)
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
        self.G_hat = fft2(create_gauss_peak(self.size, self.sigma))
        # plt.imshow(ifft2(self.G_hat).astype(float))
        # plt.show()
        self.H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)
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
        y_shift, x_shift = np.array(np.unravel_index(R.argmax(), R.shape))
        

        if x_shift > self.size[0] / 2: x_shift -= self.size[0]
        if y_shift > self.size[1] / 2: y_shift -= self.size[1]

        self.position[0] += x_shift
        self.position[1] += y_shift

        H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)

        self.H_hat = (1-self.alpha) * self.H_hat + self.alpha * H_hat

        return [self.position[0] - self.patch_size[0]/2, self.position[1] - self.patch_size[1]/2, self.patch_size[0], self.patch_size[1]]

    def set_params(self):
        self.enlarge = 1.2
        self.sigma = 2
        self.alpha = 0.02
        self.lamb = 1e-3