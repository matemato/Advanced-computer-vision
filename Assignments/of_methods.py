import cv2
import numpy as np
from ex1_utils import gaussderiv, gausssmooth, show_flow, rotate_image

def get_derivatives(im1,im2, s=1):
    Ix1, Iy1 = gaussderiv(im1, s)
    Ix2, Iy2 = gaussderiv(im2, s)
    Ix = (Ix1 + Ix2)/2
    Iy = (Iy1 + Iy2)/2
    It = gausssmooth(im2 - im1, s)

    return Ix, Iy, It

def lucas_kanade(im1, im2, N, opt=False):

    Ix, Iy, It = get_derivatives(im1, im2)
    kernel = np.ones((N,N))

    Ixt = cv2.filter2D(cv2.multiply(Ix, It), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Iyt = cv2.filter2D(cv2.multiply(Iy, It), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Ixx = cv2.filter2D(cv2.multiply(Ix, Ix), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Iyy = cv2.filter2D(cv2.multiply(Iy, Iy), -1, kernel, borderType=cv2.BORDER_REFLECT)
    Ixy = cv2.filter2D(cv2.multiply(Ix, Iy), -1, kernel, borderType=cv2.BORDER_REFLECT)

    D = cv2.multiply(Iyy,Ixx) - cv2.multiply(Ixy, Ixy) + 1e-15

    u = - (cv2.divide((cv2.multiply(Iyy, Ixt) - cv2.multiply(Ixy,Iyt)), D))
    v = - (cv2.divide((cv2.multiply(Ixx, Iyt) - cv2.multiply(Ixy,Ixt)), D))

    if opt:
        tr = Ixx + Iyy + 1e-15
        smallest_eig = D/tr
        mask = np.ma.masked_where(smallest_eig>np.percentile(smallest_eig, 75), smallest_eig).mask
        u = u*mask
        v = v*mask

    return u,v

def horn_schunck(im1, im2, n_iters, lmbd, s=1, u=None, v=None):
    n,m = im1.shape
    if u is None and v is None:
        u,v = np.zeros((n,m), np.float32), np.zeros((n,m), np.float32)
    Ld = np.array([[0,   1/4,    0],
                   [1/4, 0,      1/4],
                   [0,   1/4,    0]])
    Ix, Iy, It = get_derivatives(im1, im2, s)

    for i in range(n_iters):
        ua = cv2.filter2D(u, -1, Ld, borderType=cv2.BORDER_REFLECT)
        va = cv2.filter2D(v, -1, Ld, borderType=cv2.BORDER_REFLECT)
        P = cv2.multiply(Ix, ua) + cv2.multiply(Iy, va) + It
        D = lmbd + cv2.multiply(Ix, Ix) + cv2.multiply(Iy,Iy) + 1e-15
        u = ua - cv2.multiply(Ix, cv2.divide(P,D))
        v = va - cv2.multiply(Iy, cv2.divide(P,D))
                
    return u,v


