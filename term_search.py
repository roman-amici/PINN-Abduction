from bayes_opt import BayesianOptimization
from PINN_Base import Scalar_PDE, ScalarDifferentialTerm
from util import rmse
import numpy as np
import tensorflow as tf

def bayes_opt_validation(
    model_function,
    term_library, 
    X_train,
    U_train,
    X_eval,
    U_eval,
    reps=1,
    init_points=3,
    n_iter=8,
    alpha=5e-2):

    def evaluation_function(**kwargs):

        terms = []

        for idx, val in kwargs.items():
            if val > 0.5:
                terms.append(term_library[int(idx)])

        errors = []

        for _ in range(reps):
            model = model_function(terms)

            model.train_BFGS(X_train,U_train)
            U_hat = model.predict(X_eval)

            errors.append( rmse(U_eval,U_hat) )

        #Minimize error by maximizing negative error
        best_error = -np.min(errors)

        #clean up the model
        model.cleanup()
        tf.reset_default_graph()

        return best_error

    bounds = { str(i) : (0,1) for i in range(len(term_library)) }
    optimizer = BayesianOptimization(
        evaluation_function,
        bounds
    )

    #Some heuristics here...
    optimizer.maximize(init_points=init_points,n_iter=n_iter, alpha=alpha)

    return optimizer

def grid_search_validation(
    model_function,
    term_library, 
    X_train,
    U_train,
    X_eval,
    U_eval,
    n_trials=12,
    reps=1):

    random_choice = (np.random.random(size=(len(term_library)) ) > 0.5)

    trials = []
    for _ in range(n_trials):

        terms = []
        for i,b in enumerate(random_choice):
            if b:
                terms.append(term_library[i])

        errors = []

        for _ in range(reps):
            model = model_function(terms)

            model.train_BFGS(X_train,U_train)
            U_hat = model.predict(X_eval)

            errors.append( rmse(U_eval,U_hat) )

        #Minimize error by maximizing negative error
        best_error = np.min(errors)

