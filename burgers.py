import numpy as np
import matplotlib.pyplot as plt
import util
from PINN_Base import ScalarDifferentialTerm, Scalar_PDE
import term_search
import data_load

import argparse


def train_burgers(X_train, U_train, layers, differential_terms) -> Scalar_PDE:
    lower_bound = np.array([-1, 0])  # x,t
    upper_bound = np.array([1, 1])

    model = Scalar_PDE(
        differential_terms,
        layers,
        lower_bound,
        upper_bound,
        regularization_param=1)

    model.train_BFGS(X_train, U_train)

    return model


def evaluate_burgers(X_eval, U_eval, model: Scalar_PDE):
    U_hat = model.predict(X_eval)

    return util.rmse(U_eval, U_hat)


def burgers_model(terms, infer_params=False):
    lower_bound = np.array([-1, 0])  # x,t
    upper_bound = np.array([1, 1])

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    return Scalar_PDE(
        terms,
        layers,
        lower_bound,
        upper_bound,
        regularization_param=1,
        infer_params=infer_params)


def run_burgers(
        n_train=1000,
        n_eval=1000,
        reps=1,
        n_iter=20,
        n_init=5,
        data_noise=0.1,
        bayes_alpha=1e-2,
        infer_params=True,
        mode="nonlinear",
        du_order=2,
        u_order=1,
        search="bayes"):

    t, x, u = data_load.load_burgers()

    X, U = data_load.flatten_burgers(t, x, u)
    X_train, U_train = util.subset_data(X, U, n_train)
    # noise 10% of variation in the data
    U_train = util.percent_noise(U_train, data_noise)

    X_eval, U_eval = util.subset_data(X, U, n_eval)
    U_eval = util.percent_noise(U_eval, data_noise)

    if mode == "linear":
        term_library = ScalarDifferentialTerm.get_linear_combinations_scalar(
            2, du_order)
    else:
        term_library = ScalarDifferentialTerm.get_combinations_scalar(
            2, u_order, du_order)

    # Add the true parameter value into the library, just to see if it can find it.
    if not infer_params:
        for term in term_library:
            if term.u_order == 0 and term.du_order == 2 and term.du_component == 0:
                new_term = ScalarDifferentialTerm(0, 2, 0, -.01 / np.pi)
                break
        term_library.append(new_term)

    def burgers_model_t(terms):
        return burgers_model(terms, infer_params)

    if search == "bayes":
        optimizer = term_search.bayes_opt_validation(
            burgers_model_t,
            term_library,
            X_train, U_train,
            X_eval, U_eval,
            reps,
            n_init,
            n_iter,
            bayes_alpha)

        term_vector = optimizer.max["params"]

        best_terms = []
        for idx, val in term_vector.items():
            if val > 0.5:
                best_terms.append(term_library[int(idx)])

    elif search == "random":
        errors, terms = term_search.random_search_validation(
            burgers_model_t,
            term_library,
            X_train, U_train,
            X_eval, U_eval,
            n_iter,
            reps)

        best_error_idx = np.argmin(errors)
        best_terms = terms[best_error_idx]

    elif search == "grid":
        best_terms = term_search.grid_search_evaluation(
            burgers_model_t,
            term_library,
            X_train, U_train,
            X_eval, U_eval)

    util.print_scalar_terms(best_terms)

    model = burgers_model(best_terms, infer_params)
    model.train_BFGS(X_train, U_train)

    print("Best Error:", evaluate_burgers(X, U, model))
    print("Params:")
    print(model.get_params())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization using solutions to the burgers equation.")
    parser.add_argument('--ntrain', type=int,
                        help="Number of training points", default=1000)
    parser.add_argument("--neval", type=int,
                        help="Number of evaluation poitns", default=1000)
    parser.add_argument(
        "--nreps", type=int, help="Number of repetiions of each term for Bayesian Optimization for variance stabalization", default=1)
    parser.add_argument(
        "--niter", type=int, help="Number of iterations of Bayesian Optimization", default=25)
    parser.add_argument(
        "--ninit", type=int, help="Number of exploration iterations in Bayesian Optimization", default=3)
    parser.add_argument("--noise", type=float,
                        help="Noise level (as a percentage of the data's std) to modify the data for", default=0.1)
    parser.add_argument("--alpha", type=float,
                        help="Uncertainty in loss function for Bayesian Optimization", default=1e-2)
    parser.add_argument("--infer_params", action="store_true",
                        help="Whether to infer the parameters for each differential operator. If not, the true values will be used")
    parser.add_argument("--mode", choices=["linear", "nonlinear"],
                        help="Combinatorial scheme for the dictionary", default="nonlinear")
    parser.add_argument("--duorder", type=int,
                        help="Highest derivative order to add to the term dictionary", default=2)
    parser.add_argument(
        "--uorder", type=int, help="Highest u order to add to the term dictionary (non-linear)",  default=1)
    parser.add_argument(
        "--search", choices=["bayes", "random", "grid"], default="bayes")

    args = parser.parse_args()
    n_train = args.ntrain
    n_eval = args.neval
    reps = args.nreps
    n_iter = args.niter
    n_init = args.ninit
    data_noise = args.noise
    bayes_alpha = args.alpha
    infer_params = args.infer_params
    mode = args.mode
    du_order = args.duorder
    u_order = args.uorder
    search = args.search

    run_burgers(
        n_train,
        n_eval,
        reps,
        n_iter,
        n_init,
        data_noise,
        bayes_alpha,
        infer_params,
        mode,
        du_order,
        u_order,
        search)
