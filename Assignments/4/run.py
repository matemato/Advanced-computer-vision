import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

def random_walk(q, r=1):
    T = sp.symbols('T')
    F = sp.Matrix(np.zeros((2,2)))
    L = sp.Matrix([[1,0], [0,1]])
    Fi = sp.exp(F*T)
    Q = sp.integrate( (Fi*L)*q*(Fi*L).T, (T, 0, 1) )
    R = r * np.eye(2)
    H = np.eye(2)
    return np.array(Fi).astype('float64'), H, np.array(Q).astype('float64'), R

def NCV(q=1,r=1):
    T = sp.symbols('T')
    F = sp.Matrix(np.vstack((np.array([[0,0,1,0],[0,0,0,1]]), np.zeros((2,4)))))
    L = sp.Matrix(np.vstack((np.zeros((2,2)), np.eye(2))))
    Fi = sp.exp(F*T)
    Q = sp.integrate( (Fi*L)*q*(Fi*L).T, (T, 0, 1) )
    R = r * np.eye(2)
    H = np.array([[1,0,0,0],[0,1,0,0]])
    return np.array(sp.exp(F)).astype('float64'), H, np.array(Q).astype('float64'), R

def NCA(q,r=1):
    T = sp.symbols('T')
    F = sp.Matrix(np.diag(np.ones(4), 2))
    L = sp.Matrix(np.vstack((np.zeros((4,2)), np.eye(2))))
    # L = sp.Matrix(np.vstack((np.zeros((2,2)), np.eye(2), np.eye(2))))
    # L = sp.Matrix(np.zeros((6,2)))
    Fi = sp.exp(F*T)
    Q = sp.integrate( (Fi*L)*q*(Fi*L).T, (T, 0, 1) )
    R = r * np.eye(2)
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
    return np.array(sp.exp(F)).astype('float64'), H, np.array(Q).astype('float64'), R

if __name__ == "__main__":

    params = np.array([[0, 100, 1], [1, 5, 1], [2,1,1], [3,1,5], [4,1,100]])

    fig, ax = plt.subplots(3, 5, figsize=(18,12))
    # fig.suptitle('Horizontally stacked subplots')
    # ax[0][0].imshow(im)
    # ax[0][1].imshow(im2)
    # ax[0][2].imshow(im3)
    # ax[1][0].imshow(im4)
    # ax[1][1].imshow(im5)
    # ax[1][2].imshow(im6)

    # ax[0, 0].set_title('Non-ocluded target')
    # ax[0, 1].set_title('Partially ocluded target')
    # ax[0, 2].set_title('Mostly ocluded target')

    # ax[1, 0].set_title('PSR = 25.8')
    # ax[1, 1].set_title('PSR = 8.7')
    # ax[1, 2].set_title('PSR = 4.6')

    for i in range(3):
        for k, q, r in params:
            
                N = 40
                # v = np.linspace(5*math.pi, 0, N)
                # x = np.cos(v) * v
                # y = np.sin(v) * v

                t = np.linspace(0,2*np.pi,N)

                # heart
                x = 16*np.sin(t)**3
                y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

                # trefoil knot
                x = np.sin(t) + 2*np.sin(2*t)
                y = np.cos(t) - 2*np.cos(2*t)
                
                if i == 0: 
                    A, C, Q_i, R_i = random_walk(q,r)
                    ax[i,k].set_title(f'RW: q = {q}, r = {r}')
                elif i == 1: 
                    A, C, Q_i, R_i = NCV(q,r)
                    ax[i,k].set_title(f'NCV: q = {q}, r = {r}')
                else: 
                    A, C, Q_i, R_i = NCA(q,r)
                    ax[i,k].set_title(f'NCA: q = {q}, r = {r}')
                sx=np.zeros((x.size,1), dtype=np.float32).flatten()
                sy=np.zeros((y.size,1), dtype=np.float32).flatten()
                sx[0]=x[0]
                sy[0]=y[0]

                state=np.zeros((A.shape[0],1), dtype=np.float32).flatten()
                state[0] = x[0]
                state[1] = y[0]

                covariance=np.eye(A.shape[0], dtype=np.float32)
                for j in range(1,x.size):
                    state,covariance,_,_=kalman_step(A,C,Q_i,R_i,np.reshape(np.array([x[j],y[j]]),(-1,1)),np.reshape(state,(-1,1)),covariance)
                    sx[j]=state[0]
                    sy[j]=state[1]

                ax[i,k].plot(x,y, '-ro')
                ax[i,k].plot(sx, sy, '-bo')
    plt.show()


