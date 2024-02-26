#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import pickle
import time

import GPy
import networkx
import numpy
import paramz
from .dgp import SGHMC_DGP


class M4Rbf(GPy.kern.Kern):
    def __init__(self, input_dim, variance=None, lengthscale=None,
                 active_dims=None, name="M4Rbf", useGPU=False):

        super(M4Rbf, self).__init__(input_dim, active_dims, name,
                                    useGPU=useGPU)

        # if variance is not None:
        #    variance = numpy.asarray(variance)
        # else:
        #    variance = numpyy.array(1.)
        if lengthscale is not None and len(lengthscale) == input_dim // 2:
            lengthscale = numpy.asarray(lengthscale)
        else:
            lengthscale = numpy.ones(self.input_dim // 2)
        # self.variance = GPy.core.parameterization.param.Param("variance",
        # variance, paramz.transformations.Logexp())
        self.lengthscale = GPy.core.parameterization.param.Param("lengthscale", lengthscale, paramz.transformations.Logexp())

        # self.link_parameters(self.variance)
        self.link_parameters(self.lengthscale)

    def to_dict(self) -> dict:
        input_dict = super(M4Rbf, self)._save_to_input_dict()

        # input_dict["variance"] = self.variance.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        input_dict["class"] = "M4Rbf"

        return input_dict

    @paramz.caching.Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None) -> numpy.ndarray:
        if X2 is None:
            X2 = X

        m1, n1 = X.shape
        m2, n2 = X2.shape
        assert n1 == n2 and n1 // 2 * 2 == n1
        n = n1 // 2

        mus1 = X[:, :n]
        vars1 = X[:, n:]
        mus2 = X2[:, :n]
        vars2 = X2[:, n:]

        mu = numpy.array([mus1[:, j, None] - mus2[:, j, None].T for j in range(n)])
        var = numpy.array([vars1[:, j, None] + vars2[:, j, None].T for j in range(n)])

        musq = numpy.square(mu)
        varplsq = var.copy()
        for k in range(n):
            varplsq[k, :, :] += self.lengthscale[k]**2

        # return self.variance * (numpy.prod(self.lengthscale) / numpy.sqrt(numpy.prod(varplsq, axis=0))) * numpy.exp(-.5 * numpy.sum(musq / varplsq, axis=0))
        return numpy.prod(self.lengthscale) / \
            numpy.sqrt(numpy.prod(varplsq, axis=0)) * \
            numpy.exp(-.5 * numpy.sum(musq / varplsq, axis=0))

    def Kdiag(self, X) -> numpy.ndarray:
        return numpy.diag(self.K(X))

    def reset_gradients(self) -> None:
        # self.variance.gradient = numpy.zeros(1)
        self.lengthscale.gradient = numpy.zeros(self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X) -> None:
        # self.variance.gradient = 0.
        self.lengthscale.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None, reset=True):
        X2 = X if X2 is None else X2

        m1, n1 = X.shape
        m2, n2 = X2.shape
        assert n1 == n2 and n1 // 2 * 2 == n1
        n = n1 // 2

        mus1 = X[:, :n]
        vars1 = X[:, n:]
        mus2 = X2[:, :n]
        vars2 = X2[:, n:]

        mu = numpy.array([mus1[:, j, None] - mus2[:, j, None].T for j in range(n)])
        var = numpy.array([vars1[:, j, None] + vars2[:, j, None].T for j in range(n)])

        musq = numpy.square(mu)
        varplsq = var.copy()

        for k in range(n):
            varplsq[k, :, :] += self.lengthscale[k]**2

        K = (numpy.prod(self.lengthscale) / numpy.sqrt(numpy.prod(varplsq, axis=0))) * numpy.exp(-.5 * numpy.sum(musq / varplsq, axis=0))

        tmp = dL_dK * K
        grads = n * [0]  # create placeholder
        for k in range(n):
            KK = 1.0 / self.lengthscale[k] - self.lengthscale[k] / varplsq[k] \
                 + self.lengthscale[k] * mu[k] / varplsq[k]**2
            grads[k] = numpy.sum(tmp * KK)

#       self.variance.gradient = numpy.sum(tmp * K) / self.variance
        self.lengthscale.gradient = grads


class M4Model(object):

    def __init__(self, X={}, Y={}, M={}, G=None):
        self.X = X
        self.Y = Y
        self.M = M
        self.G = G

    @staticmethod
    def load(filename):
        return pickle.load(open(filename, "rb"))

    def dump(self, filename) -> None:
        return pickle.dump(self, open(filename, "wb"))

    def predict(self, x, fidelity, muvar=None, **kwargs):
        lev, fid = fidelity

        if muvar is None:
            muvar = {}

        ancestors = list(networkx.ancestors(self.G, fidelity)) + [fidelity]
        ancestors.sort(key=lambda lvfid: lvfid[0] * 3 + ((lvfid[1] + 2) % 3))
        for ans in ancestors:
            if ans not in muvar:
                preds = list(self.G.predecessors(ans))
                preds.sort(key=lambda levfid: levfid[0] * 3 + ((levfid[1] + 2) % 3))
                xx = numpy.concatenate((x, *[muvar[pred][0] for pred in preds]), axis=1)
                if kwargs["model.propagate_var"][ans[1]]:
                    xx = numpy.concatenate((xx, *[muvar[pred][1] for pred in preds]), axis=1)
                muvar[ans] = self.M[ans].predict(xx)

        return muvar

    def train_submodel(self, fidelity, **kwargs) -> None:
        t1 = time.time()
        lev, fid = fidelity

        # build enhanced X
        X = self.X[fidelity]
        input_dim = X.shape[1]

        preds = list(self.G.predecessors(fidelity))
        preds.sort(key=lambda levfid: levfid[0] * 3 + ((levfid[1] + 2) % 3))
        muvar = {}
        for pred in preds:
            if pred not in muvar:
                muvar = self.predict(X, pred, muvar, **kwargs)
        nf = len(preds)
        X = numpy.concatenate((X, *[muvar[pred][0] for pred in preds]), axis=1)
        if kwargs["model.propagate_var"][fid]:
            X = numpy.concatenate((X, *[muvar[pred][1] for pred in preds]), axis=1)
            nf *= 2

        # build and/or train model
        if kwargs["model.type"][fidelity[1]] == "GP":
            if fidelity not in self.M:
                ker = m4kernel(input_dim, nf, linear=kwargs["model.linear"],
                               nonlinear=kwargs["model.nonlinear"],
                               ARD=kwargs["model.ARD"])
                self.M[fidelity] = GPy.models.GPRegression(X, self.Y[fidelity], kernel=ker, Y_metadata=None, normalizer=None, noise_var=None)

#           self.M[fidelity][".*lengthscale"].constrain_bounded(1e-3, 1e3)
            self.M[fidelity][".*Gaussian_noise"] = self.M[fidelity].Y.var() * 0.01
            self.M[fidelity][".*Gaussian_noise"].fix()
            self.M[fidelity].optimize(max_iters=500)
            self.M[fidelity][".*Gaussian_noise"].unfix()
#           self.M[fidelity][".*Gaussian_noise"].constrain_positive()
            self.M[fidelity][".*Gaussian_noise.variance"].constrain_bounded(1e-12, 1e-6)

            self.M[fidelity].optimize_restarts(
                num_restarts=kwargs["model.restarts"],
                robust=True,
                verbose=kwargs["model.verbose"],
                paralle=(kwargs["model.threads"] is not None and kwargs["model.threads"] > 1),
                num_processes=kwargs["model.threads"],
                messages=kwargs["model.verbose"],
                optimizer="bfgs",
                start=None,
                max_iters=kwargs["model.max_iters"],
                ipython_notebook=False,
                clear_after_finish=True
            )

        elif kwargs["model.type"][fidelity[1]] == "DGP":
            if fidelity not in self.M:
                self.M[fidelity] = SGHMC_DGP(
                    X,
                    self.Y[fidelity],
                    n_inducing=len(X),
                    n_layers=kwargs["model.n_layers"],
                    iterations=kwargs["model.max_iters"],
                    ncores=kwargs["model.threads"]
                )

            self.M[fidelity].fit(X, self.Y[fidelity])

        else:
            raise ValueError("model.type can be either 'GP' or 'DGP'")

        t2 = time.time()
        print("Train time: ", (t2 - t1))

    def train(self, **kwargs) -> None:
        nodes = list(self.G.nodes)
        nodes.sort(key=lambda levfid: levfid[0] * 3 + ((levfid[1] + 2) % 3))
        for fidelity in nodes:
            self.train_submodel(fidelity, **kwargs)


def m4kernel(X_dim, nf, linear=False, nonlinear=True, ARD=False):
    input_dim = X_dim + nf

    if nf == 0 or not (linear or nonlinear):
        k0 = GPy.kern.RBF(input_dim, variance=1., lengthscale=None, ARD=ARD,
                          active_dims=None, name="rbf", useGPU=False,
                          inv_l=False)

    else:
        input_dims = list(range(X_dim))
        multifid_dims = list(range(X_dim, X_dim + nf))

        k3 = GPy.kern.RBF(X_dim, variance=1., lengthscale=None, ARD=ARD,
                          active_dims=input_dims, name="rbf", useGPU=False,
                          inv_l=False)
        if linear:
            k5 = GPy.kern.Linear(nf//2, variances=None, ARD=ARD, active_dims=multifid_dims[:nf//2], name="linear")
#           k5 = M4Linear(nf, variances=None, ARD=ARD, active_dims=multifid_dims, name="M4Linear")
            k4 = k5
        if nonlinear:
            k6 = M4Rbf(nf, lengthscale=None, active_dims=multifid_dims,
                       name="M4Rbf", useGPU=False)
            k4 = k6
        if linear and nonlinear:
            k4 = GPy.kern.Add((k5, k6), name="sum")
        k1 = GPy.kern.Prod((k3, k4), name="mul")
        k2 = GPy.kern.RBF(X_dim, variance=1., lengthscale=None, ARD=ARD, active_dims=input_dims, name="rbf", useGPU=False, inv_l=False)
        k0 = GPy.kern.Add((k1, k2), name="sum")

    return k0
