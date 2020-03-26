import numpy as np
from scipy.io import loadmat
from os.path import exists
from collections.abc import Iterable

# Assumes equal spacing


def to_grid_scalar(X, U, n_0, n_1):
    assert(X.shape[1] == 2)
    assert(U.shape[1] == 1)

    min_0, max_0 = np.min(X[:, 0]), np.max(X[:, 0])
    min_1, max_1 = np.min(X[:, 1]), np.max(X[:, 1])

    range_0 = (max_0 - min_0)
    range_1 = (max_1 - min_1)

    grid_0 = np.zeros((n_0, n_1))
    grid_1 = np.zeros((n_0, n_1))
    u_grid = np.zeros((n_0, n_1))

    for i in range(X.shape[0]):
        x_0, x_1 = X[i, 0], X[i, 1]
        c_0 = int(((x_0/range_0) - (min_0/range_0))*n_0)
        c_1 = int(((x_1/range_1) - (min_1/range_1))*n_1)

        u_grid[c_0, c_1] = U[i, 0]
        grid_0[c_0, c_1] = x_0
        grid_1[c_0, c_1] = x_1

    return u_grid, grid_0, grid_1


def subset_data(X, U, n):
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sub = X[idx, :]
    U_sub = U[idx, :]

    return X_sub, U_sub


def rmse(U, U_hat):
    return np.sqrt(np.mean((U_hat[:, 0] - U[:, 0])**2))


def percent_noise(U, noise_percent=0.1):
    std = np.std(U[:, 0])*noise_percent
    return U + np.random.normal(0, std, size=U.shape)


def print_scalar_terms(scalar_terms):
    s = ""
    for term in scalar_terms:
        s += term.__repr__() + " + "
    return s


def term_vecotr_to_sdt(term_vector, term_library):
    terms = []
    for idx, val in term_vector.items():
        if val > 0.5:
            terms.append(term_library[int(idx)])

    return terms


def term_dict_extent(max_u_order, max_du_order):
    convention = ["x", "t"]

    if isinstance(max_u_order, Iterable) and isinstance(max_du_order, Iterable):
        s = ""
        for i, (u_order, du_order) in enumerate(zip(max_u_order, max_du_order)):
            s += f"u^{u_order} du_{convention[i]}{du_order} + "
        return s
    else:
        return f"u^{max_u_order} du_{max_du_order}"


def compare_term_lists(tl1, tl2):
    if len(tl1) != len(tl2):
        return False
    else:
        for t1, t2 in zip(tl1, tl2):
            if t1 != t2:
                return False
        return True


def correct_solution_searched(correct_terms, optimizer, term_library):
    search_list = optimizer.space.res()

    for search_point in search_list:
        params = search_point["params"]
        term_vector = term_vecotr_to_sdt(params, term_library)
        if compare_term_lists(term_vector, correct_terms):
            return True

    return False


def log_optimizer_run(filepath, run_id, optimizer):
    search_list = optimizer.space.res()

    with open(f"{filepath}/run_{run_id}.csv", "w+") as f:

        f.write("target,params\n")

        for search_point in search_list:
            target = search_point["target"]
            params = search_point["params"]

            params = ",".join([str(val) for val in params.values()])

            f.write(f"{target},{params}\n")


def log_trial(filepath, **kwargs):
    columns = ["run_id", "PDE", "search_method",
               "n_train", "n_eval", "data_noise", "infer_params",
               "best_solution", "solution_correct", "correct_solution_searched",
               "n_init", "n_iter", "dictionary_extent", "kernel", "acquisition_function", "alpha", "kappa", "xi",
               "eval_error", "test_error"]

    c_set = set(columns)
    for key in kwargs:
        if key not in c_set:
            print(f"WARNING: {key} not logged")

    if not exists(filepath):
        with open(filepath, "w+") as f:
            f.write(",".join(columns) + "\n")

    row = []
    for key in columns:
        if key in kwargs:
            row.append(str(kwargs[key]))
        else:
            row.append("")

    line = ",".join(row) + "\n"
    with open(filepath, "a") as f:
        f.write(line)
