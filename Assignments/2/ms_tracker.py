import numpy as np
import cv2
import matplotlib . pyplot as plt

from ex2_utils import Tracker, get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram


class MSTracker(Tracker):
    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = [int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)]
        region[2] = int(region[2])
        region[3] = int(region[3])
        if region[2]%2==0: region[2] -= 1
        if region[3]%2==0: region[3] -= 1
        self.size = (region[2], region[3])
        self.X, self.Y = np.meshgrid(np.arange(-int(self.size[0]/2), int(self.size[0]/2) + 1), np.arange(-int(self.size[1]/2), int(self.size[1]/2) + 1))
        self.ep = create_epanechnik_kernel(self.size[0], self.size[1], 1)
        patch,_ = get_patch(image, self.position, self.size)
        self.q = extract_histogram(patch, 16, self.ep)
        self.q = self.q/np.sum(self.q)

        plt.imshow(image)
        plt.plot(self.position[0], self.position[1], 'rx')
        plt.show()


    def track(self, image):
        for i in range(20):
            patch,_ = get_patch(image, self.position, self.size)
            p = extract_histogram(patch, 16, self.ep)
            p = p/np.sum(p)

            backproject = backproject_histogram(patch, np.sqrt(self.q/(p+1e-3)),16)
            backproject = backproject/np.sum(backproject)

            self.position[0] = self.position[0] + np.sum(self.X*backproject)
            self.position[1] = self.position[1] + np.sum(self.Y*backproject)

        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2