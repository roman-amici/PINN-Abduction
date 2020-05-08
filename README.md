# PINN-Abduction

# Approach

This repository contains code based on a (failed) project for using [Physics Informed Neural Networks](https://maziarraissi.github.io/PINNs/) in order to infer the "best" differential equation which explains a data set. It is thus an example of logic [Abduction](https://en.wikipedia.org/wiki/Abductive_reasoning) using PINNs (in case you were wondering about the name). It was motivated by the fact that PINN based style regularization using the correct equation and parameters tends produces error values which dominate all other neural network solutions which attempt to explain the data (giving sufficient noise).

The approach taken was to train a large number of PINNs, each with a different underlying differential equation, using the PINN with the lowest error as the inferred differential equation. Even for fairly modest assumptions, the space of solutions grows like 2^N where N is the number of terms in our library. Thus, grid search is intractable. Instead, we construct a surrogate model of the error function and use the surrogate to suggest promising points to search at. At first, we used [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization). However, Bayesian Optimization (as it is usually formulated) requires that we are optimizing over real valued parameters instead of the binary variables we use to select the inclusion of a differential term. We switched to [SMAC3](https://github.com/automl/SMAC3) which uses random forests to deal with binary variables.

Using SMAC, we saw minor success. If the parameters of the differential equation to test over were already known, then the correct solution could reliably be recovered after searching 10%-20% of the total search space. However, when asked to jointly infer the parameters, it consistently failed to find the correct solution even though it had tried the correct solution during the course of its search. Thus appears to be too much noise in the process to distinguish the correct solution from spurious solutions when the parameters must also be inferred.

If you happen to find yourself in the unlikely situation where you are trying to distinguish between a number of fully parameterized differential equations, perhaps this code might be useful to you.

# Installation

You will need Tensorflow Version 1 to run this. It will not work with Tensorflow 2.0 since the newer version has dropped support for L-BFGS. You will also need either [BayesOptimization](https://github.com/fmfn/BayesianOptimization) or [SMAC3](https://github.com/automl/SMAC3).

# Usage

See example.ipynb for an example run using [Google Colab](https://colab.research.google.com/). Note that the BayesianOptimization portions of the code have been neglected and are not as fully featured as the SMAC code paths.
