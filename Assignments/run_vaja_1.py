import cv2
import time
import numpy as np
import matplotlib . pyplot as plt
from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck

def example():
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    print(type(im1))
    show_both(im1, im2)

def test():
    # im1 = cv2.imread('collision/00000181.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    im1 = cv2.imread('disparity/office2_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    # im2 = cv2.imread('collision/00000182.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    im2 = cv2.imread('disparity/office2_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

    # im1 = cv2.imread('custom/ezgif-frame-075.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    # im2 = cv2.imread('custom/ezgif-frame-076.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

    show_both(im1, im2)

def time_test():
    # im1 = cv2.imread('collision/00000001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    im1 = cv2.imread('disparity/office2_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    # im2 = cv2.imread('collision/00000002.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    im2 = cv2.imread('disparity/office2_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    # im1 = cv2.imread('lab2/001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    # im2 = cv2.imread('lab2/002.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    start = time.time()
    U_lk , V_lk = lucas_kanade(im1, im2, 3)
    end = time.time()
    print(end - start)
    U_hs , V_hs = horn_schunck(im1, im2, 1000, 0.5)
    start = time.time()
    U_hs , V_hs = horn_schunck(im1, im2, 1000, 0.5)
    end = time.time()
    print(end - start)

    start = time.time()
    # U_hs , V_hs = horn_schunck(im1, im2, 1000, 0.5)
    U_hs2 , V_hs2 = horn_schunck(im1, im2, 500, 0.5, U_lk, V_lk)
    end = time.time()
    print(end - start)

    show(im1, im2, U_lk, V_lk, 'Lucas−Kanade Optical Flow')
    show(im1, im2, U_hs, V_hs, 'Horn−Schunck Optical Flow ')
    show(im1, im2, U_hs2, V_hs2, 'Horn−Schunck Optical Flow Initialized With Optimized')

def show(im1, im2, u, v, suptitle):
    fig, ((ax1_11, ax1_12), (ax1_21 , ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(u ,v, ax1_21, type='angle')
    show_flow(u, v, ax1_22, type='field', set_aspect=True)
    fig.suptitle(suptitle)
    plt.show()

def show_both(im1, im2):
    U_lk , V_lk = lucas_kanade(im1, im2, 3)
    U_hs , V_hs = horn_schunck(im1, im2, 1000, 0.5)
    fig1, ((ax1_11, ax1_12), (ax1_21 , ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk ,V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle ('Lucas−Kanade Optical Flow Optimized')
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type='angle' )
    show_flow (U_hs, V_hs, ax2_22, type='field', set_aspect=True )
    fig2.suptitle('Horn−Schunck Optical Flow ')
    plt.show()

if __name__ == "__main__":
    # example()
    # test()
    time_test()



