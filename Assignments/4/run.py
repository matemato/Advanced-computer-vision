import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

def random_walk(q, r=1, w=10, h=10):
    alpha = 0.1
    beta = 0.5
    T = sp.symbols('T')
    F = sp.Matrix(np.zeros((2,2)))
    L = sp.Matrix([[1,0], [0,1]])
    Fi = sp.exp(F*T)
    Q = sp.integrate( (Fi*L)*q*(Fi*L).T, (T, 0, 1) )
    R = r * np.eye(2)
    H = np.eye(2)
    P = np.array([[alpha*w,0,0,0], [0,alpha*h,0,0], [0,0,beta*w,0], [0,0,0,beta*h]])

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

    N = 40
    v = np.linspace(5*math.pi, 0, N)
    x = np.cos (v) * v
    y = np.sin (v) * v
    vx = np.diff(x)
    vx = np.insert(vx, 0, 0)
    vy = np.diff(y)
    vy = np.insert(vy, 0, 0)
    ax = np.diff(vx)
    ax = np.insert(ax, 0, 0)
    ay = np.diff(vy)
    ay = np.insert(ay, 0, 0)
    
    A, C, Q_i, R_i = random_walk(1,100)
    A, C, Q_i, R_i = NCV(1,100)
    A, C, Q_i, R_i = NCA(1,100)
    sx=np.zeros((x.size,1), dtype=np.float32).flatten()
    sy=np.zeros((y.size,1), dtype=np.float32).flatten()
    sx[0]=x[0]
    sy[0]=y[0]
    svx=np.zeros((vx.size,1), dtype=np.float32).flatten()
    svy=np.zeros((vy.size,1), dtype=np.float32).flatten()
    svx[0]=vx[0]
    svy[0]=vy[0]
    sax=np.zeros((ax.size,1), dtype=np.float32).flatten()
    say=np.zeros((ay.size,1), dtype=np.float32).flatten()
    sax[0]=ax[0]
    say[0]=ay[0]
    state=np.zeros((A.shape[0],1), dtype=np.float32).flatten()
    state[0]=x[0]
    state[1]=y[0]
    state[2] = vx[0]
    state[3] = vx[1]
    covariance=np.eye(A.shape[0], dtype=np.float32)
    for j in range(1,x.size):
        state,covariance,_,_=kalman_step(A,C,Q_i,R_i,np.reshape(np.array([x[j],y[j]]),(-1,1)),np.reshape(state,(-1,1)),covariance)
        sx[j]=state[0]
        sy[j]=state[1]
        svx[j]=state[2]
        svy[j]=state[3]
        sax[j]=state[4]
        say[j]=state[5]

    plt.plot(x,y)
    plt.plot(sx, sy)
    plt.show()

    plt.plot(vx,vy)
    plt.plot(svx, svy)
    plt.show()
    plt.plot(ax,ay)
    plt.plot(sax, say)
    plt.show()


