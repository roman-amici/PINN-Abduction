import numpy as np
import matplotlib.pyplot as plt
import util
from PINN_Base import ScalarDifferentialTerm, Scalar_PDE

def train_burgers(X_train,U_train,layers,differential_terms) -> Scalar_PDE:
    lower_bound = np.array([-1,0]) #x,t
    upper_bound = np.array([1,1])

    model = Scalar_PDE(
        differential_terms, 
        layers, 
        lower_bound,
        upper_bound,
        regularization_param=1)

    model.train_BFGS(X_train,U_train)

    return model

def evaluate_burgers(X_eval,U_eval,model : Scalar_PDE):
    U_hat = model.predict(X_eval)

    return util.rmse(U_eval,U_hat)

t,x,u = util.load_burgers()
nt = t.shape[1]
nx = x.shape[1]

X,U = util.flatten_burgers(t,x,u)
X_train,U_train = util.subset_data(X,U,2000)

burgers_equation = [
    ScalarDifferentialTerm(0,1,1), #u_t
    ScalarDifferentialTerm(1,1,0), #u u_x
    ScalarDifferentialTerm(0,2,0) #u_xx
]

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = train_burgers(X_train, U_train, layers, burgers_equation)

print(evaluate_burgers(X,U,model))