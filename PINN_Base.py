import tensorflow as tf
import numpy as np
from tensorflow.contrib.opt import ScipyOptimizerInterface
from typing import List

class PINN:

    def __init__(
        self,
        layers : List[int],
        lower_bound : np.array,
        upper_bound : np.array,
        regularization_param=1):

        self.layers = layers
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.regularization_param = regularization_param

    def _build_net(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.weights,self.biases = self.__init_NN(self.layers)
            
            self.X = tf.placeholder(tf.float32, shape=[None,self.layers[0]])
            self.U = tf.placeholder(tf.float32, shape=[None,self.layers[-1]])

            self.U_hat = self.__NN(self.X)
            self.loss_U = self._get_loss(self.U,self.U_hat)
            self.loss_dU = self._get_loss_du()

            self.loss = self.loss_U + self.loss_dU

            self.optimizer_BFGS = ScipyOptimizerInterface(
                self.loss,
                method = 'L-BFGS-B', 
                options = {'maxiter': 50000,
                            'maxfun': 50000,
                            'maxcor': 50,
                            'maxls': 50,
                            'ftol' : 1.0 * np.finfo(float).eps})
            
            init = tf.global_variables_initializer()

        self.sess = tf.Session(
            graph=self.graph, 
            config=tf.ConfigProto(allow_soft_placement=True))

        self.sess.run(init)

    def _get_loss(self,U,U_hat):
        return tf.reduce_mean( tf.square( U-U_hat ) )

    def _get_loss_du(self):
        return 0.0

    def __init_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.__xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def __xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))

        return tf.Variable(tf.truncated_normal(
            [in_dim, out_dim], 
            stddev=xavier_stddev), dtype=tf.float32)

    def __NN(self,X):
        Z = 2.0*(X - self.lower_bound)/(self.upper_bound - self.lower_bound) - 1.0

        for l in range(len(self.weights)-1):
            W = self.weights[l]
            b = self.biases[l]
            Z = tf.tanh(tf.add(tf.matmul(Z, W), b))

        W = self.weights[-1]
        b = self.biases[-1]
        U = tf.add(tf.matmul(Z, W), b)

        return U

    def train_BFGS(self,X,U):
        self.optimizer_BFGS.minimize(self.sess, {self.X : X, self.U : U})

