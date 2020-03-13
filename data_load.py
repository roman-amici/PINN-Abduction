import numpy as np
from scipy.io import loadmat

def load_burgers(path="data/mat-data/burgers_shock.mat"):

    data = loadmat(path)

    t = data['t']
    x = data['x']
    u = np.real(data['usol'])

    return t,x,u

def flatten_2D(x1,x2,u):
    n_x1 = x1.shape[0]
    n_x2 = x2.shape[0]

    X = np.zeros((n_x1*n_x2, 2))
    U = np.zeros((n_x1*n_x2, 1))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            idx = i*n_x2 + j
            X[idx,0] = x1[i]
            X[idx,1] = x2[j]
            U[idx,0] = u[i,j]

    return X,U

def flatten_burgers(t,x,u):
  n_x = x.shape[0]
  n_t = t.shape[0]
  X = np.zeros((n_x*n_t, 2))
  U = np.zeros((n_x*n_t,1))
  for x_i in range(x.shape[0]):
    for t_i in range(t.shape[0]):
      idx = x_i*n_t + t_i
      X[idx,0] = x[x_i]
      X[idx,1] = t[t_i]
      U[idx,0] = u[x_i,t_i]

  return X,U

def load_kdv(path="data/mat-data/KdV.mat"):
    data = loadmat(path)

    t = data['tt']
    x = data['x']
    u = np.real(data['uu'])

    return t,x,u

def flatten_kdv(t,x,u):
    n_x = x.shape[0]
    n_t = t.shape[0]

    X = np.zeros((n_x*n_t, 2))
    U = np.zeros((n_x*n_t,1))

    for x_i in range(x.shape[0]):
        for t_i in range(t.shape[0]):
            idx = x_i*n_t + t_i
            X[idx,0] = x[x_i]
            X[idx,1] = t[t_i]
            U[idx,0] = u[x_i,t_i]

    return X,U

def load_helmholtz(path="data/npy-data"):
    x1 = np.load(f"{path}/helmholtz-x1.npy")
    x2 = np.load(f"{path}/helmholtz-x2.npy")
    U = np.load(f"{path}/helmholtz-U.npy")

    return x1,x2,U
