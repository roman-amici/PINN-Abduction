import numpy as np
import matplotlib.pyplot as plt
import util
from PINN_Base import ScalarDifferentialTerm, Scalar_PDE
import term_search
import data_load
import tensorflow as tf
from collections.abc import Iterable

import argparse

NU = -.01 / np.pi

burgers_true = [
    ScalarDifferentialTerm(0, 1, 1),  # u_t
    ScalarDifferentialTerm(1, 1, 0),  # u u_x
    ScalarDifferentialTerm(0, 2, 0, NU),  # u_xx
]


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


def burgers_model(terms, regularization, infer_params=False):
    lower_bound = np.array([-1, 0])  # x,t
    upper_bound = np.array([1, 1])

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    return Scalar_PDE(
        terms,
        layers,
        lower_bound,
        upper_bound,
        regularization_param=regularization,
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
        search="bayes",
        kappa=2.576,
        xi=0.0,
        log_file="",
        run_log_directory="",
        acquisition_function="ucb",
        use_regularization=False,
        noisy_eval=True,
        random_param_init=False):

    t, x, u = data_load.load_burgers()

    X, U = data_load.flatten_burgers(t, x, u)
    X_train, U_train = util.subset_data(X, U, n_train)
    # noise 10% of variation in the data
    U_train = util.percent_noise(U_train, data_noise)

    X_eval, U_eval = util.subset_data(X, U, n_eval)
    if noisy_eval:
        U_eval = util.percent_noise(U_eval, data_noise)

    if mode == "linear":
        term_library = ScalarDifferentialTerm.get_linear_combinations_scalar(
            2, du_order)
    else:
        if isinstance(u_order, Iterable) and isinstance(du_order, Iterable):
            term_library = ScalarDifferentialTerm.no_cross_combinations(
                list(u_order), list(du_order))
        else:
            term_library = ScalarDifferentialTerm.get_combinations_scalar(
                2, u_order, du_order)

    # Add the true parameter value into the library, just to see if it can find it.
    if not infer_params:
        for i, term in enumerate(term_library):
            if term.u_order == 0 and term.du_order == 2 and term.du_component == 0:
                new_term = ScalarDifferentialTerm(0, 2, 0, NU)
                term_library[i] = new_term
                break
    elif random_param_init:
        for term in term_library:
            term.param = np.random.randn()
    elif infer_params:
        for term in term_library:
            # Default initialization of zero for inference only.
            term.param = 0.0

    def burgers_model_t(terms, regularization=1.0):
        return burgers_model(terms, regularization, infer_params)

    if search == "bayes":
        optimizer = term_search.bayes_opt_validation(
            burgers_model_t,
            term_library,
            X_train, U_train,
            X_eval, U_eval,
            reps,
            n_init,
            n_iter,
            bayes_alpha,
            acquisition_function,
            kappa,
            xi)

        best_terms = util.term_vecotr_to_sdt(
            optimizer.max["params"], term_library)

        eval_error = optimizer.max["target"]
        solution_correct, searched_solution = util.compare_term_lists(
            burgers_true, best_terms)
        correct_solution_searched = util.correct_solution_searched(
            burgers_true, optimizer, term_library)

        tf.reset_default_graph()
        model = burgers_model(best_terms, infer_params)
        model.train_BFGS(X_train, U_train)

        test_error = evaluate_burgers(X, U, model)
        run_id = np.random.randint(0, 2**63-1, dtype=np.int64)
        if log_file:
            util.log_trial(
                log_file,
                run_id=int(run_id),
                PDE="Burgers",
                search_method=search,
                n_train=n_train,
                n_eval=n_eval,
                data_noise=data_noise,
                infer_params=infer_params,
                best_solution=util.print_scalar_terms(best_terms),
                solution_correct=solution_correct,
                correct_solution_searched=correct_solution_searched,
                alpha=bayes_alpha,
                n_init=n_init,
                n_iter=n_iter,
                dictionary_extent=util.term_dict_extent(
                    u_order, du_order),
                kernel="matern-2.5",
                acquisition_function=acquisition_function,
                kappa=kappa,
                xi=xi,
                eval_error=eval_error,
                test_error=test_error,
            )

        return optimizer, run_id

    elif search == "smac":
        smac_result, incumbent = term_search.smac_validation(
            burgers_model_t,
            term_library,
            X_train, U_train,
            X_eval, U_eval,
            n_iter,
            use_regularization
        )

        run_history = smac_result.get_runhistory()
        eval_error = run_history.get_cost(incumbent)
        best_terms = util.smac_config_to_sdt(incumbent, term_library)
        best_regularization = incumbent["reg"] if use_regularization else 1.0

        solution_correct = util.compare_term_lists(burgers_true, best_terms)
        correct_solution_searched, searched_solution = util.smac_correct_solution_searched(
            burgers_true, run_history, term_library)

        model = burgers_model(best_terms, best_regularization, infer_params)
        model.train_BFGS(X_train, U_train)

        test_error = evaluate_burgers(X, U, model)

        run_id = np.random.randint(0, 2**63-1, dtype=np.int64)
        if log_file:
            util.log_trial(
                log_file,
                run_id=int(run_id),
                PDE="Burgers",
                search_method=search,
                n_train=n_train,
                n_eval=n_eval,
                data_noise=data_noise,
                infer_params=infer_params,
                best_solution=util.print_scalar_terms(best_terms),
                solution_correct=solution_correct,
                correct_solution_searched=correct_solution_searched,
                n_iter=n_iter,
                dictionary_extent=util.term_dict_extent(
                    u_order, du_order),
                eval_error=eval_error,
                test_error=test_error,
                regularization_search=use_regularization,
                best_regularization=best_regularization,
                noisy_eval=noisy_eval,
                random_param_init=random_param_init,
                best_solution_with_params=util.print_scalar_terms_with_params(
                    best_terms),
                searched_solution_with_params=util.print_scalar_terms_with_params(
                    searched_solution) if searched_solution is not None else ""
            )

        if run_log_directory:
            util.log_run_smac(run_log_directory, run_id,
                              term_library, run_history)

        return run_history, run_id

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization using solutions to the burgers equation.")
    parser.add_argument('--ntrain', type=int,
                        help="Number of training points", default=1000)
    parser.add_argument("--neval", type=int,
                        help="Number of evaluation points", default=1000)
    parser.add_argument(
        "--nreps", type=int, help="Number of repetitions of each term for Bayesian Optimization for variance stabalization", default=1)
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
        "--search", choices=["smac", "bayes", "random", "grid"], default="bayes")
    parser.add_argument("--logfile", type=str, default="",
                        help="Path to the file to log each trial in")
    parser.add_argument("--acq", choices=["ucb", "ei", "poi"], default="ucb",
                        help="Acquisition function for Bayesian Optimization")
    parser.add_argument("--kappa", type=float, default=2.576,
                        help="Exploration parameter")
    parser.add_argument("--xi", type=float, default=0.0,
                        help="Exploitation Parameter")
    parser.add_argument("--reg_search", action="store_true",
                        help="Perform search for regularization strength jointly with the terms. (SMAC only)", default=False)
    parser.add_argument("--pristine_eval", action="store_true",
                        help="Evaluate error function on an evaluation set without added noise")
    parser.add_argument("--random_param_init", action="store_true",
                        help="Randomly initialize parameters to the differential equation (otherwise initial values are zero)")
    use_regularization = False,
    noisy_eval = True,
    random_param_init = False

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
    logfile = args.logfile
    acq = args.acq
    kappa = args.kappa
    xi = args.xi
    use_regularization = args.reg_search
    noisy_eval = not args.pristine_eval
    random_param_init = args.random_param_init

    run_burgers(
        n_train=n_train,
        n_eval=n_eval,
        reps=reps,
        n_iter=n_iter,
        n_init=n_init,
        data_noise=data_noise,
        bayes_alpha=bayes_alpha,
        infer_params=infer_params,
        mode=mode,
        du_order=du_order,
        u_order=u_order,
        search=search,
        log_file=logfile,
        acquisition_function=acq,
        kappa=kappa,
        xi=xi,
        use_regularization=use_regularization,
        noisy_eval=noisy_eval,
        random_param_init=random_param_init)
