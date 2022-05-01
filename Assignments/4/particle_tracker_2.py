import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
import cv2

from ex4_utils import sample_gauss
from ex3_utils import create_cosine_window, create_gauss_peak
from run_vaja_4 import NCV, random_walk, NCA
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, Tracker
# from utils.tracker import Tracker

class ParticleTracker(Tracker):
    def __init__(self, q=0.01, motion=1, nam="0"):
        self.bins = 16
        self.sigma = 4
        self.alpha = 0.01
        self.n = 70
        self.q_percantage = q
        self.nam = nam
        self.motion = motion
        self.lamb = 1e-3
        np.random.seed(0)

    def name(self):
        return self.nam

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        self.q = np.mean(self.size)*self.q_percantage
        if self.motion == 0: self.Fi, _, self.Q, _ = random_walk(q=self.q)
        elif self.motion == 1: self.Fi, _, self.Q, _ = NCV(q=self.q)
        else: self.Fi, _, self.Q, _ = NCA(q=self.q)
                
        state = np.zeros(len(self.Fi))
        state[0], state[1]  = self.position[0], self.position[1]
        self.X = (sample_gauss(np.zeros(len(self.Fi)), self.Q, self.n) + state[np.newaxis,:]) # used to be @ self.Fi.T here
        self.W = np.ones(self.n) / self.n

        self.win = create_cosine_window(self.size)
        p,_ = get_patch(image, self.position, self.size)


        F_hat = fft2(self.win * p)
        self.G_hat = fft2(create_gauss_peak(self.size, 1))
        self.H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)

        return self.X[:,:2], self.W
         
    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,w = image.shape
        weights_cumsumed=np.cumsum(self.W) # cumulative distribution
        rand_samples=np.random.rand(self.n,1)
        sampled_idxs=np.digitize(rand_samples,weights_cumsumed) # randomly select N indices
        self.X = self.X[sampled_idxs.flatten(),:] @ self.Fi.T + sample_gauss(np.zeros(len(self.Fi)), self.Q, self.n)

        for i, x in enumerate(self.X):
            if x[0]>w or x[1]>h or x[0]<0 or x[1]<0:
                continue
            patch,_ = get_patch(image, (x[0], x[1]), self.size)
            F_hat = fft2(self.win * patch)
            R = ifft2(self.H_hat * F_hat)
            self.W[i] = self.get_PSR(R) * np.max(R.real)**2


        self.W /= np.sum(self.W)
        
        self.position = self.W @ self.X[:,:2]

        p,_ = get_patch(image, self.position, self.size)
        F_hat = fft2(self.win * p)

        H_hat = (self.G_hat * np.conj(F_hat)) / (F_hat * np.conj(F_hat) + self.lamb)

        if 'R' in locals():
            self.H_hat = (1-self.alpha) * self.H_hat + self.alpha * H_hat      
        
        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]], self.X[:,:2], self.W
        # return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]

    def get_PSR(self, R):
            G = np.roll(R.real, (int(self.size[1]/2), int(self.size[0]/2)), (0, 1))

            y, x = np.array(np.unravel_index(G.argmax(), G.shape))
            mask = np.ones([self.size[1], self.size[0]]).astype('uint8')
            left = max(0, x-11)
            top = max(0, y-11)
            right = min(x+12, self.size[0])
            bottom = min(y+12, self.size[1])
            mask[top:bottom, left:right] = 0
            mask = np.ma.make_mask(mask)

            PSR = ( np.max(G) - np.mean(G[mask]) ) / np.std(G[mask])
            # print(PSR)
            return PSR

def hellinger(p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


        
        
        

if __name__ == "__main__":
    np.random.seed(0)
    Pt = ParticleTracker()
    Pt.initialize()

    
    