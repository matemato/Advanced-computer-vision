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
        self.size = [region[2], region[3]]
        self.X, self.Y = np.meshgrid(np.arange(-int(self.size[0]/2), int(self.size[0]/2) + 1), np.arange(-int(self.size[1]/2), int(self.size[1]/2) + 1))
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        patch,_ = get_patch(image, self.position, self.size)
        self.q = extract_histogram(patch, self.parameters.bins, self.kernel)

        self.c = 1
        self.c = self.get_background_info(image)
        
        self.q = self.c * self.q
        self.q = self.q/np.sum(self.q)
        # plt.imshow(image)
        # plt.plot(self.position[0], self.position[1], 'rx')
        # plt.show()


    def track(self, image):
        for i in range(20):
            patch,_ = get_patch(image, self.position, self.size)

            # plt.imshow(patch)
            # plt.plot(self.position[0], self.position[1], 'rx')
            # plt.show()

            p = extract_histogram(patch, self.parameters.bins, self.kernel) * self.c
            p = p/np.sum(p)

            backproject = backproject_histogram(patch, np.sqrt(self.q/(p+self.parameters.eps)), self.parameters.bins)
            backproject = backproject/np.sum(backproject)

            x_shift = np.sum(self.X*backproject)
            y_shift = np.sum(self.Y*backproject)
            self.position[0] += x_shift
            self.position[1] += y_shift

            if int(np.sqrt(x_shift**2 + y_shift**2)) <= self.parameters.stop_dist:
                break

        # plt.imshow(patch)
        # plt.plot(self.position[0], self.position[1], 'rx')
        # plt.show()
        
        self.update_q(patch)
        
        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]

    def get_background_info(self, image):
        background_patch,_ = get_patch(image, self.position, np.array(self.size)*3)
        background_hist = extract_histogram(background_patch, self.parameters.bins, create_epanechnik_kernel(self.size[0]*3, self.size[1]*3, self.parameters.sigma)) - self.q

        o_min = min(background_hist[background_hist>0])
        background_hist = [max(i, 0) for i in background_hist]
        return [min(o_min/i, 1) for i in background_hist]

    def update_q(self, patch):
        new_q = extract_histogram(patch, self.parameters.bins, self.kernel) * self.c
        new_q = new_q/np.sum(new_q)
        self.q = (1-self.parameters.alpha) * self.q + self.parameters.alpha * new_q


class MSParams():
    def __init__(self):
        self.stop_dist = 1
        self.bins = 16
        self.eps = 1e-2
        self.sigma = 0.5
        self.alpha = 0.05