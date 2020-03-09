import numpy as np
from scipy.io import loadmat

#Assumes equal spacing
def to_grid_scalar(X,U,n_0,n_1):
    assert(X.shape[1] == 2)
    assert(U.shape[1] == 1)

    min_0,max_0 = np.min(X[:,0]), np.max(X[:,0])
    min_1,max_1 = np.min(X[:,1]), np.max(X[:,1])

    range_0 = (max_0 - min_0)
    range_1 = (max_1 - min_1)

    grid_0 = np.zeros( (n_0,n_1) )
    grid_1 = np.zeros( (n_0,n_1))
    u_grid = np.zeros( (n_0,n_1) )

    for i in range(X.shape[0]):
        x_0,x_1 = X[i,0],X[i,1]
        c_0 = int(((x_0/range_0) - (min_0/range_0))*n_0)
        c_1 = int(((x_1/range_1) - (min_1/range_1))*n_1)

        u_grid[c_0,c_1] = U[i,0]
        grid_0[c_0,c_1] = x_0
        grid_1[c_0,c_1] = x_1

    return u_grid,grid_0,grid_1

def load_burgers(path="data/mat-data/burgers_shock.mat"):

    data = loadmat(path)

    t = data['t']
    x = data['x']
    u = np.real(data['usol'])

    return t,x,u

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

def subset_data(X,U,n):
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sub = X[idx,:]
    U_sub = U[idx,:]

    return X_sub,U_sub

def rmse(U,U_hat):
    return np.sqrt(np.mean( (U_hat[:,0] - U[:,0])**2))