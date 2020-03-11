import numpy as np
import matplotlib.pyplot as plt
import util
from PINN_Base import ScalarDifferentialTerm, Scalar_PDE
import term_search

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

def burgers_model(terms):
    lower_bound = np.array([-1,0]) #x,t
    upper_bound = np.array([1,1])

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    return Scalar_PDE(
        terms, 
        layers, 
        lower_bound,
        upper_bound,
        regularization_param=1)    

t,x,u = util.load_burgers()
nt = t.shape[1]
nx = x.shape[1]

X,U = util.flatten_burgers(t,x,u)
X_train,U_train = util.subset_data(X,U,1000)
U_train = util.percent_noise(U_train,0.1)#noise 10% of variation in the data

X_eval,U_eval = util.subset_data(X,U,1000)
U_eval = util.percent_noise(U_eval,0.1)

#Up to second order in u and nonlinear (like u^2)
term_library = ScalarDifferentialTerm.get_combinations_scalar(2,1,2)

optimizer = term_search.bayes_opt_validation(
    burgers_model, 
    term_library,
    X_train,U_train,
    X_eval, U_eval,1,5,25)

term_vector = optimizer.max["params"]

best_terms = []
for idx,val in term_vector.items():
    if val > 0.5:
        best_terms.append(term_library[int(idx)])

s = ""
for term in best_terms:
    s += term.__repr__() + " + "
print(s)

model = burgers_model(best_terms)
model.train_BFGS(X_train,U_train)

print("Best Error:", evaluate_burgers(X,U,model))
print("Params:")
print(model.get_params())