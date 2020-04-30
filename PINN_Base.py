import tensorflow as tf
import numpy as np
from tensorflow.contrib.opt import ScipyOptimizerInterface
from typing import List
from collections import namedtuple


class PINN:

    def __init__(
            self,
            layers: List[int],
            lower_bound: np.array,
            upper_bound: np.array,
            dtype=tf.float32,
            regularization_param=1.0):

        self.layers = layers
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dtype = dtype
        self.regularization_param = regularization_param

        self._build_net()

    def cleanup(self):
        self.sess.close()

        del self.sess

    def _build_net(self):

        with tf.Graph().as_default() as g:

            self._init_variables()

            self._init_placeholders()

            self.U_hat = self.__NN(self.X)
            self.loss_U = self._get_loss(self.U, self.U_hat)
            self.loss_dU = self._get_loss_du()

            self.loss = self.loss_U + self.regularization_param * self.loss_dU

            self.optimizer_BFGS = ScipyOptimizerInterface(
                self.loss,
                method='L-BFGS-B',
                options={'maxiter': 50000,
                         'maxfun': 50000,
                         'maxcor': 50,
                         'maxls': 50,
                         'ftol': 1.0 * np.finfo(float).eps,
                         'gtol': 1.0 * np.finfo(float).eps})

            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=g)

        self.sess.run(init)
        self.sess.graph.finalize()

    def _init_variables(self):
        self.weights, self.biases = self.__init_NN(self.layers)

    def _init_placeholders(self):
        self.X = tf.placeholder(self.dtype, shape=[None, self.layers[0]])
        self.U = tf.placeholder(self.dtype, shape=[None, self.layers[-1]])

    def _get_loss(self, U, U_hat):
        return tf.reduce_mean(tf.square(U - U_hat))

    def _get_loss_du(self):
        return 0.0

    def __init_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.__xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(
                tf.zeros([1, layers[l+1]], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def __xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))

        return tf.Variable(tf.truncated_normal(
            [in_dim, out_dim],
            stddev=xavier_stddev), dtype=self.dtype)

    def __NN(self, X):
        Z = 2.0*(X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0

        for l in range(len(self.weights)-1):
            W = self.weights[l]
            b = self.biases[l]
            Z = tf.tanh(tf.add(tf.matmul(Z, W), b))

        W = self.weights[-1]
        b = self.biases[-1]
        U = tf.add(tf.matmul(Z, W), b)

        return U

    def train_BFGS(self, X, U):
        self.optimizer_BFGS.minimize(self.sess, {self.X: X, self.U: U})

    def predict(self, X):
        return self.sess.run(self.U_hat, {self.X: X})


class ScalarDifferentialTerm:

    def __init__(self,
                 u_order: int,
                 du_order: int,
                 du_component: int,
                 param=1.0):
        # Param will be the default value if parameters are inferred,
        # and the only value if parameters are not inferred.

        self.u_order = u_order
        self.du_order = du_order
        self.du_component = du_component
        self.param = param

        if du_order == 0 and du_component != 0:
            raise Exception(
                "Invalid value du_component when u_order is du_order is 0")

    def __eq__(self, other):
        return (self.u_order == other.u_order) and \
            (self.du_order == other.du_order) and \
            (self.du_component == other.du_component)

    def __repr__(self):

        convention = ["x", "t"]

        if self.param == 1:
            p_str = ""
        else:
            p_str = str(self.param)

        if self.u_order == 0:
            u_str = ""
        elif self.u_order > 1:
            u_str = f"u^{self.u_order}"
        else:
            u_str = "u"

        if self.du_order == 0:
            du_str = ""
        else:
            du_str = "u_"
            for _ in range(self.du_order):
                du_str += convention[self.du_component]

        return u_str + " " + du_str

    @staticmethod
    def get_combinations_scalar(
            n_components: int,
            max_u_order: int,
            max_du_order: int) -> List:

        combinations = []
        for du_order in range(max_du_order+1):
            for u_order in range(max_u_order+1):
                if u_order == 0 and du_order == 0:
                    continue
                elif du_order == 0:  # No order for just u**n since its a scalar
                    combinations.append(
                        ScalarDifferentialTerm(u_order, du_order, 0))
                else:
                    for component in range(n_components):
                        combinations.append(ScalarDifferentialTerm(
                            u_order, du_order, component))

        return combinations

    @staticmethod
    def no_cross_combinations(
            max_u_order: List[int],
            max_du_order: List[int]) -> List:

        assert(len(max_u_order) == len(max_du_order))
        n_components = len(max_u_order)

        combinations = []
        combinations.append(ScalarDifferentialTerm(1, 0, 0))

        for component in range(n_components):
            for u_order in range(max_u_order[component]+1):
                for du_order in range(1, max_du_order[component]+1):
                    combinations.append(
                        ScalarDifferentialTerm(u_order, du_order, component)
                    )

        return combinations

    @staticmethod
    def get_linear_combinations_scalar(
            n_components: int,
            max_du_order: int) -> List:

        combinations = [ScalarDifferentialTerm(0, 1, 0)]
        for du_order in range(1, max_du_order):
            for du_component in range(n_components):
                combinations.append(ScalarDifferentialTerm(
                    0, du_order, du_component))

        return combinations


class Scalar_PDE(PINN):

    def __init__(
            self,
            differential_terms: List[ScalarDifferentialTerm],
            layers: List[int],
            lower_bound: np.array,
            upper_bound: np.array,
            dtype=tf.float32,
            regularization_param=1,
            infer_params=True,
            forcing_function=None):

        self.differential_terms = differential_terms
        self.forcing_function = forcing_function
        self.infer_params = infer_params

        self.max_differential_order = 0
        for term in differential_terms:
            self.max_differential_order = max(
                self.max_differential_order, term.du_order)

        super().__init__(layers, lower_bound, upper_bound, dtype, regularization_param)

    def _init_variables(self):
        super()._init_variables()

        self.differential_params = []
        for i, term in enumerate(self.differential_terms):
            if i == 0 or not self.infer_params:
                # The only have n-1 linearly independent parameters, by convention we simply fix the first one
                param = tf.constant(term.param, dtype=self.dtype)
            else:
                param = tf.Variable(term.param, dtype=self.dtype)

            self.differential_params.append(param)

    def _get_loss_du(self):

        partials = self.__get_partial_derivatives(
            self.max_differential_order, self.U_hat, self.X)

        loss = 0.0
        for i, term in enumerate(self.differential_terms):

            tensor = self.__term_to_tensor(
                term,
                self.differential_params[i],
                self.U_hat,
                partials)
            loss += tensor

        if self.forcing_function:
            loss += -self.forcing_function(self.X)

        return tf.reduce_mean(tf.square(loss))

    def __term_to_tensor(self,
                         term: ScalarDifferentialTerm,
                         term_param: tf.Variable,
                         U: tf.Tensor,
                         partials: List[List[tf.Tensor]]):

        if term.u_order == 0:
            u_n = 1.0
        else:
            u_n = (U[:, 0])**term.u_order

        if term.du_order == 0:
            return term_param * u_n
        else:
            return term_param * u_n * \
                partials[term.du_order-1][term.du_component]

    def __get_partial_derivatives(self, max_order, U, X) -> List[List[tf.Tensor]]:

        if max_order == 0:
            return []

        du_1 = tf.gradients(U, X)[0]
        du_Xi = [du_1[:, i] for i in range(du_1.shape[1])]
        partial_derivatives = [du_Xi]
        for d in range(1, max_order):
            du = partial_derivatives[d-1]
            partials = []
            for i in range(X.shape[1]):
                du_dXi = tf.gradients(du[i], X)[0]
                partials.append(du_dXi[:, i])
            partial_derivatives.append(partials)

        return partial_derivatives

    def get_params(self):
        return self.sess.run(self.differential_params)
