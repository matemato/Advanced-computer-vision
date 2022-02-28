import cv2
import numpy as np
from ex1_utils import gaussderiv, gausssmooth, show_flow, rotate_image

def lucas_kanade(im1, im2, N):

    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = (Ix1 + Ix2)/2
    Iy = (Iy1 + Iy2)/2
    It = gausssmooth(im2 - im1, 1)
    kernel = np.ones((N,N))

    Ixt = cv2.filter2D(cv2.multiply(Ix, It), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Iyt = cv2.filter2D(cv2.multiply(Iy, It), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Ixx = cv2.filter2D(cv2.multiply(Ix, Ix), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Iyy = cv2.filter2D(cv2.multiply(Iy, Iy), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Ixy = cv2.filter2D(cv2.multiply(Ix, Iy), -1, kernel, borderType=cv2.BORDER_REFLECT)

    D = cv2.multiply(Iyy,Ixx) - cv2.multiply(Ixy, Ixy)

    u = - (cv2.divide((cv2.multiply(Iyy, Ixt) - cv2.multiply(Ixy,Iyt)), D))
    v = - (cv2.divide((cv2.multiply(Ixx, Iyt) - cv2.multiply(Ixy,Ixt)), D))

    return u,v

def horn_schunck(im1 , im2 , n_iters , lmbd):
    pass


