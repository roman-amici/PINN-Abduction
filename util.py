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

def subset_data(X,U,n):
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sub = X[idx,:]
    U_sub = U[idx,:]

    return X_sub,U_sub

def rmse(U,U_hat):
    return np.sqrt(np.mean( (U_hat[:,0] - U[:,0])**2))

def percent_noise(U,noise_percent=0.1):
  std = np.std(U[:,0])*noise_percent
  return U + np.random.normal(0,std,size=U.shape)

def print_scalar_terms(scalar_terms):
    s = ""
    for term in scalar_terms:
        s += term.__repr__() + " + "
    print(s)