
from bayes_opt import BayesianOptimization
from PINN_Base import Scalar_PDE, ScalarDifferentialTerm
from util import rmse, print_scalar_terms
import numpy as np
import tensorflow as tf

from itertools import combinations

from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

import time


def smac_validation(
        model_function,
        term_library,
        X_train,
        U_train,
        X_eval,
        U_eval,
        n_iter):

    def evaluation_function(params: dict, instance, budget, **kwargs):
        terms = []

        for idx in range(len(term_library)):
            if params[str(idx)] > 0.5:
                terms.append(term_library[int(idx)])

        errors = []

        model = model_function(terms)

        model.train_BFGS(X_train, U_train)
        U_hat = model.predict(X_eval)

        errors.append(rmse(U_eval, U_hat))

        model.cleanup()

        # Minimize
        best_error = np.min(errors)

        return best_error

    terms = []
    for i in range(len(term_library)):
        terms.append(
            UniformIntegerHyperparameter(str(i), 0, 1,)
        )

    cs = ConfigurationSpace()
    cs.add_hyperparameters(terms)

    scenario = Scenario({
        "run_obj": "quality",
        "cs": cs,
        "runcount-limit": n_iter,
        "limit_resources": False,
        "deterministic": True,
    })

    smac = SMAC4HPO(
        scenario=scenario,
        tae_runner=evaluation_function
    )

    incumbent = smac.optimize()

    return smac, incumbent


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
        alpha=5e-2,
        acq="ucb",
        kappa=2.765,
        xi=0.0):

    def evaluation_function(**kwargs):

        terms = []

        for idx, val in kwargs.items():
            if val > 0.5:
                terms.append(term_library[int(idx)])

        errors = []

        for _ in range(reps):
            model = model_function(terms)

            model.train_BFGS(X_train, U_train)

            U_hat = model.predict(X_eval)

            errors.append(rmse(U_eval, U_hat))

            model.cleanup()

        # Minimize error by maximizing negative error
        best_error = -np.min(errors)

        return best_error

    bounds = {str(i): (0, 1) for i in range(len(term_library))}
    optimizer = BayesianOptimization(
        evaluation_function,
        bounds,
    )

    # Some heuristics here...
    optimizer.maximize(init_points=init_points,
                       n_iter=n_iter,
                       acq=acq,
                       kappa=kappa,
                       xi=xi,
                       alpha=alpha)

    return optimizer


def random_search_validation(
        model_function,
        term_library,
        X_train,
        U_train,
        X_eval,
        U_eval,
        n_trials=12,
        reps=1):

    best_error = np.inf

    trial_term = []
    trial_error = []
    for t in range(n_trials):

        random_choice = (np.random.random(size=(len(term_library))) > 0.5)

        terms = []
        for i, b in enumerate(random_choice):
            if b:
                terms.append(term_library[i])

        errors = []
        for _ in range(reps):
            model = model_function(terms)

            model.train_BFGS(X_train, U_train)
            U_hat = model.predict(X_eval)

            errors.append(rmse(U_eval, U_hat))

        best_rep_error = np.min(errors)

        if best_rep_error < best_error:
            best_error = best_rep_error
            star = "*"

        print(t)
        print_scalar_terms(terms)
        print(best_rep_error, star)

        trial_term.append(terms)
        trial_error.append(best_rep_error)

        model.cleanup()

    return trial_error, trial_term


def grid_search_evaluation(
        model_function,
        term_library,
        X_train,
        U_train,
        X_eval,
        U_eval):

    # WARNING: Grows like 2**len(term_library)

    t = 0

    best = np.inf
    best_terms = []
    for r in range(1, len(term_library)):
        for combo in combinations(r, term_library):
            model = model_function(combo)
            model.train_BFGS(X_train, U_train)

            error = rmse(U_eval, model.predict(X_eval))
            star = ""
            if error < best:
                best = error
                best_terms = combo
                star = "*"

            print(t)
            print_scalar_terms(combo)
            print(error, star)
            t += 1

    return best_terms
