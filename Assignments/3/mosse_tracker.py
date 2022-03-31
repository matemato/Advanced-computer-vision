from pickletools import uint8
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

class MOSSETracker(Tracker):
    def __init__(self):
        self.enlarge = 1
        self.sigma = 1
        self.alpha = 0.06
        # self.alpha = 0.6
        self.lamb = 1e-3
        # self.scales = [0.9, 1, 1.1]
        self.scales = [1]
        self.training_images = 8

    def name(self):
        return 'MOSSE_Tracker_SCALE'

    def initialize(self, image, region):
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
        self.G_hat = fft2(create_gauss_peak(self.size, self.sigma))
 
        self.get_training_set(image)
        
        # self.H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)
        # plt.imshow(ifft2(self.H_hat).astype(float))
        # plt.show()

        # plt.imshow(ifft2(F_hat * self.H_hat).astype(float))
        # plt.show()


    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # p,_ = get_patch(image, self.position, self.size)

        # F_hat = fft2(self.win * p)

        # plt.imshow(ifft2(F_hat).astype(float))
        # plt.show()

        # R = ifft2(self.H_hat * F_hat)

        # plt.imshow(R.astype(float))
        # plt.show()

        F_hat, R = self.get_best_size(image)

        y_shift, x_shift = np.array(np.unravel_index(R.argmax(), R.shape))
        
        if x_shift > self.size[0] / 2: x_shift -= self.size[0]
        if y_shift > self.size[1] / 2: y_shift -= self.size[1]

        self.position[0] += x_shift
        self.position[1] += y_shift

        if(self.update(R)):

            self.A = self.alpha * self.G_hat * np.conj(F_hat) + (1-self.alpha) * self.A
            self.B = self.alpha * F_hat * np.conj(F_hat) + (1-self.alpha) * self.B

            self.H_hat = self.A / (self.B + self.lamb)
        # H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)

        # self.H_hat = (1-self.alpha) * self.H_hat + self.alpha * H_hat

        return [self.position[0] - self.patch_size[0]/2, self.position[1] - self.patch_size[1]/2, self.patch_size[0], self.patch_size[1]]

    def get_best_size(self, image):
        max_response = 0
        for s in self.scales:
            if self.patch_size[0]*s > self.size[0]*2: continue
            p,_ = get_patch(image, self.position, self.patch_size*s)
            p = np.log(p/255+1)
            p = ( p - np.mean(p) ) / np.std(p)
            p = cv2.resize(p, self.size)
            F_hat = fft2(self.win * p)
            R = ifft2(self.H_hat * F_hat)
            response = np.max(R)

            # plt.imshow(ifft2(F_hat).astype(float))
            # plt.show()
            # plt.imshow(R.astype(float))
            # plt.show()

            if response > max_response:
                max_response = response
                best_F_hat = F_hat
                best_R = R
                best_scale = s
        self.patch_size = self.patch_size*best_scale
        return best_F_hat, best_R

    def get_training_set(self, image):
        self.A = 0
        self.B = 0
        rand = random.Random(0)
        p,_ = get_patch(image, self.position, self.size)
        p = np.log(p/255+1)
        p = ( p - np.mean(p) ) / np.std(p)
        F_hat = fft2(self.win * p)
        self.A += self.G_hat * np.conj(F_hat) # add original image to dataset
        self.B += F_hat * np.conj(F_hat)
        if self.training_images > 0:
            for i in range(self.training_images):
                rotate_matrix = cv2.getRotationMatrix2D(center=self.size/2, angle=rand.uniform(-10, 10), scale=rand.uniform(0.9, 1.1))
                rotated_image = cv2.warpAffine(src=p, M=rotate_matrix, dsize=self.size)
                # plt.imshow(rotated_image)
                # plt.show()
                F_hat = fft2(self.win * rotated_image)
                self.A += self.G_hat * np.conj(F_hat)
                self.B += F_hat * np.conj(F_hat)
        self.H_hat = self.A / (self.B + self.lamb)

    def update(self, R):
        G = np.roll(R.real, (int(self.size[1]/2), int(self.size[0]/2)), (0, 1))
        # plt.imshow(G)
        # plt.show()
        y, x = np.array(np.unravel_index(G.argmax(), G.shape))
        mask = np.ones([self.size[1], self.size[0]]).astype('uint8')
        left = max(0, x-11)
        top = max(0, y-11)
        right = min(x+12, self.size[0])
        bottom = min(y+12, self.size[1])
        mask[top:bottom, left:right] = 0
        mask = np.ma.make_mask(mask)
        # plt.imshow(mask)
        # plt.show()

        PSR = ( np.max(G) - np.mean(G[mask]) ) / np.std(G[mask])
        # print(PSR)
        return True if PSR > 10 else False