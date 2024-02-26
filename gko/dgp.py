#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import pickle

import numpy
import scipy
import tensorflow


F64 = tensorflow.float64


class Layer(object):
    def __init__(self, input_dim, output_dim, n_inducing, variance, lengthscales, fixed_mean, X):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.M = n_inducing

        # radial basis function (RBF) or squared exponential kernel
        self.variance = tensorflow.exp(tensorflow.Variable(numpy.log(variance), dtype=F64, name='log_variance'))
        lengthscales = lengthscales * numpy.ones(self.input_dim, dtype=float)
        self.lengthscales = tensorflow.exp(tensorflow.Variable(numpy.log(lengthscales), dtype=F64, name='log_lengthscales'))

        self.fixed_mean = fixed_mean

        self.Z = tensorflow.Variable(scipy.cluster.vq.kmeans2(X, self.M, minit='points')[0], dtype=F64, name='Z')
        self.Kmm = self.K(self.Z) + tensorflow.eye(self.M, dtype=F64) * 1e-7
        self.Lm = tensorflow.cholesky(self.Kmm)

        if self.input_dim == output_dim:
            self.mean = numpy.eye(self.input_dim)
        elif self.input_dim < self.output_dim:
            self.mean = numpy.concatenate([numpy.eye(self.input_dim), numpy.zeros((self.input_dim, self.output_dim - self.input_dim))], axis=1)
        else:
            _, _, V = numpy.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.output_dim, :].T

        self.U = tensorflow.Variable(numpy.zeros((self.M, self.output_dim)), dtype=F64, trainable=False, name='U')

    def to_dict(self):
        return dict(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            M=self.M,
            variance=self.variance.numpy(),
            lengthscales=self.lengthscales.numpy(),
            fixed_mean=self.fixed_mean,
            mean=self.mean,
            Z=self.Z.numpy(),
        )

    @staticmethod
    def from_dict(d):

        self = Layer(d['input_dim'], d['output_dim'], d['M'], d['variance'], d['lengthscales'], d['fixed_mean'], d['mean'])
        self.Z = d['Z']
        self.Z = tensorflow.Variable(d['Z'], dtype=F64, name='Z')
        self.Kmm = self.K(self.Z) + tensorflow.eye(self.M, dtype=F64) * 1e-7
        self.Lm = tensorflow.cholesky(self.Kmm)

        return self

    def prior(self):

        return -tensorflow.reduce_sum(tensorflow.square(self.U)) / 2.0

    def Kdiag(self, X):

        # The radial basis function (RBF) or squared exponential kernel

        return tensorflow.fill(tensorflow.shape(X)[:-1], tensorflow.squeeze(self.variance))

    def K(self, X, X2=None):

        # The radial basis function (RBF) or squared exponential kernel

        X = X / self.lengthscales
        Xs = tensorflow.reduce_sum(tensorflow.square(X), axis=-1, keepdims=True)

        if (X2 is None):
            X2 = X
            X2s = Xs
        else:
            X2 = X2 / self.lengthscales
            X2s = tensorflow.reduce_sum(tensorflow.square(X2), axis=-1, keepdims=True)

        dist = Xs + tensorflow.matrix_transpose(X2s) - 2 * tensorflow.matmul(X, X2, transpose_b=True)

        return self.variance * tensorflow.exp(-dist / 2.)

    def conditional(self, X):

        num_data = tensorflow.shape(self.Z)[0]  # M
        num_func = tensorflow.shape(self.U)[1]  # R

        # caching the covariance matrix from the sghmc steps gives a
        # significant speedup but is not being done here
        # Kmm = self.K(self.Z) + tensorflow.eye(num_data, dtype=F64) * 1e-7
        Kmm = self.Kmm
        Kmn = self.K(self.Z, X)
        Knn = self.Kdiag(X)

        # compute kernel stuff
        # Lm = tensorflow.cholesky(Kmm)
        Lm = self.Lm

        # Compute the projection matrix A
        A = tensorflow.matrix_triangular_solve(Lm, Kmn, lower=True)

        # construct the conditional mean
        mean = tensorflow.matmul(A, self.U, transpose_a=True) # N x R

        # compute the covariance due to the conditioning
        var = Knn - tensorflow.reduce_sum(tensorflow.square(A), 0)
        var = tensorflow.tile(var[None, :], [num_func, 1])  # R x N
        var = tensorflow.transpose(var)  # N x R

        if (self.fixed_mean):
            mean += tensorflow.matmul(X, tensorflow.cast(self.mean, F64))

        return mean, var


class SGHMC_DGP:
    def __init__(self, X, Y, n_inducing=100, n_layers=2, iterations=10000,
                 minibatch_size=10000, window_size=100,
                 num_posterior_samples=100, posterior_sample_spacing=50,
                 adam_lr=0.01, epsilon=0.01, mdecay=0.05, ncores=40):
        self.n_inducing = n_inducing
        self.n_layers = n_layers
        self.iterations = iterations
        self.minibatch_size = minibatch_size
        self.window_size = window_size
        self.num_posterior_samples = num_posterior_samples
        self.posterior_sample_spacing = posterior_sample_spacing
        self.adam_lr = adam_lr
        self.epsilon = epsilon
        self.mdecay = mdecay

        # data
        self.input_size = X.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.X = X
        self.Y = Y

        # batch
        self.data_iter = 0
        self.X_placeholder = tensorflow.placeholder(F64, shape=[None, self.input_dim])
        self.Y_placeholder = tensorflow.placeholder(F64, shape=[None, self.output_dim])

        # layers
        self.layers = []
        X_running = X.copy()
        for il in range(self.n_layers):
            output_dim = self.input_dim if (il+1 < self.n_layers) else self.output_dim
            layer = Layer(self.input_dim, output_dim, self.n_inducing, variance=0.01, lengthscales=float(self.input_dim)**0.5, fixed_mean=(il+1 < self.n_layers), X=X_running)
            self.layers.append(layer)
            X_running = numpy.matmul(X_running, layer.mean)

        # Parameters
        self.vars = [lyr.U for lyr in self.layers]

        # Prior
        self.prior = tensorflow.add_n([lyr.prior() for lyr in self.layers])

        # Likelihood
        self.Gaussian_likelihood_variance = tensorflow.exp(tensorflow.Variable(numpy.log(numpy.var(Y, 0)), dtype=F64, name='lik_log_variance'))

        # Inference
        self.f, self.f_means, self.f_vars, self.y_mean, self.y_var = self.propagate(self.X_placeholder)

        # Loss
        self.log_likelihood = -0.5 * (numpy.log(2 * numpy.pi) + tensorflow.log(self.y_var) + tensorflow.square(self.y_mean - self.Y_placeholder) / (self.y_var))
        self.nll = - tensorflow.reduce_sum(self.log_likelihood) / tensorflow.cast(tensorflow.shape(self.X_placeholder)[0], F64) - (self.prior / self.input_size)

        # Posterior
        self.posterior_samples = []

        # Window
        self.window = []

        self.burn_in_op, self.sample_op = self.generate_update_step(self.nll)
        self.adam = tensorflow.train.AdamOptimizer(self.adam_lr)
        self.hyper_train_op = self.adam.minimize(self.nll)

        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = ncores
        config.inter_op_parallelism_threads = ncores
        self.session = tensorflow.Session(config=config)
        init_op = tensorflow.global_variables_initializer()
        self.session.run(init_op)

    def save(self, filename):

        raise Exception("Not implemented yet")

        saver = tensorflow.train.Saver()
        pickle.dump([list(p.values()) for p in self.M.model.posterior_samples], open('aaa', 'wb'))
        y_mean = tensorflow.identity(self.M.model.y_mean, name="y_mean")
        y_var = tensorflow.identity(self.M.model.y_var, name="y_var")
        init_op = tensorflow.global_variables_initializer()
        self.M.model.session.run(init_op)
        # m, v = self.M.model.session.run((y_mean, y_var), feed_dict=feed_dict)
        saver.save(self.M.model.session, 'mydgpmodel')

    @staticmethod
    def restore(filename):

        raise Exception("Not implemented yet")

    def reset(self, X, Y):

        self.X = X
        self.Y = Y
        self.input_size = X.shape[0]
        self.data_iter = 0

    def get_minibatch(self):

        if (self.input_size <= self.minibatch_size):

            return self.X, self.Y

        else:

            if (self.data_iter > self.input_size - self.minibatch_size):

                shuffle = numpy.random.permutation(self.input_size)
                self.X = self.X[shuffle, :]
                self.Y = self.Y[shuffle, :]
                self.data_iter = 0

            X_batch = self.X[self.data_iter:self.data_iter + self.minibatch_size, :]
            Y_batch = self.Y[self.data_iter:self.data_iter + self.minibatch_size, :]
            self.data_iter += min(self.minibatch_size, self.input_size)

            return X_batch, Y_batch

    def propagate(self, X):

        Fs = [X]
        F_means, F_vars = [], []

        for layer in self.layers:
            mean, var = layer.conditional(Fs[-1])
            eps = tensorflow.random_normal(tensorflow.shape(mean), dtype=F64)
            F = mean + eps * tensorflow.sqrt(var)
            Fs.append(F)
            F_means.append(mean)
            F_vars.append(var)

        Y_mean = tensorflow.identity(F_means[-1])
        Y_var = F_vars[-1] + self.Gaussian_likelihood_variance

        return Fs[1:], F_means, F_vars, Y_mean, Y_var

    def generate_update_step(self, nll):

        burn_in_updates = []
        sample_updates = []

        grads = tensorflow.gradients(nll, self.vars)

        for theta, grad in zip(self.vars, grads):
            xi = tensorflow.Variable(tensorflow.ones_like(theta), dtype=F64, trainable=False)
            g = tensorflow.Variable(tensorflow.ones_like(theta), dtype=F64, trainable=False)
            g2 = tensorflow.Variable(tensorflow.ones_like(theta), dtype=F64, trainable=False)
            p = tensorflow.Variable(tensorflow.zeros_like(theta), dtype=F64, trainable=False)

            r_t = 1.0 / (xi + 1.0)
            g_t = (1.0 - r_t) * g + r_t * grad
            g2_t = (1.0 - r_t) * g2 + r_t * grad ** 2
            xi_t = 1.0 + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1.0 / (tensorflow.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = self.epsilon / tensorflow.sqrt(tensorflow.cast(self.input_size, F64))
            noise_scale = 2. * epsilon_scaled ** 2 * self.mdecay * Minv
            sigma = tensorflow.sqrt(tensorflow.maximum(noise_scale, 1e-16))
            sample_t = tensorflow.random_normal(tensorflow.shape(theta), dtype=F64) * sigma
            p_t = p - self.epsilon ** 2 * Minv * grad - self.mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

        burn_in_op = [tensorflow.assign(var, var_t) for var, var_t in burn_in_updates + sample_updates]
        sample_op = [tensorflow.assign(var, var_t) for var, var_t in sample_updates]

        return burn_in_op, sample_op

    def sghmc_step(self):

        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        self.session.run(self.burn_in_op, feed_dict=feed_dict)
        values = self.session.run((self.vars))
        sample = {}
        for var, value in zip(self.vars, values):
            sample[var] = value
        self.window.append(sample)
        if (len(self.window) > self.window_size):
            self.window = self.window[-self.window_size:]

    def train_hypers(self):

        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        i = numpy.random.randint(len(self.window))
        feed_dict.update(self.window[i])
        self.session.run(self.hyper_train_op, feed_dict=feed_dict)

    def print_sample_performance(self, posterior=False):

        X_batch, Y_batch = self.get_minibatch()
        feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
        if (posterior):
            feed_dict.update(numpy.random.choice(self.posterior_samples))
        mll = numpy.mean(self.session.run((self.log_likelihood), feed_dict=feed_dict), 0)
        print(' Training MLL of a sample: {}'.format(mll))

    def collect_samples(self, num, spacing):

        self.posterior_samples = []
        for i in range(num):
            for j in range(spacing):
                X_batch, Y_batch = self.get_minibatch()
                feed_dict = {self.X_placeholder: X_batch, self.Y_placeholder: Y_batch}
                self.session.run((self.sample_op), feed_dict=feed_dict)

            values = self.session.run((self.vars))
            sample = {}
            for var, value in zip(self.vars, values):
                sample[var] = value
            self.posterior_samples.append(sample)

    def fit(self, X, Y, **kwargs):

        if (len(Y.shape) == 1):
            Y = Y[:, None]

        self.reset(X, Y)

        try:
            for _ in range(self.iterations):
                self.sghmc_step()
                self.train_hypers()
                if (_ % 100 == 1):
                    print('Iteration {}'.format(_))
                    self.print_sample_performance()
            self.collect_samples(self.num_posterior_samples, self.posterior_sample_spacing)

        except KeyboardInterrupt:  # pragma: no cover
            pass

    def _predict(self, X, S):

        assert S <= len(self.posterior_samples)

        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)

        return (numpy.stack(ms, 0), numpy.stack(vs, 0))

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.num_posterior_samples)
        m = numpy.average(ms, 0)
        v = numpy.average(vs + ms**2, 0) - m**2
        return m, v

    def calculate_density(self, Xs, Ys):
        ms, vs = self._predict(Xs, self.num_posterior_samples)
        logps = scipy.stats.norm.logpdf(numpy.repeat(Ys[None, :, :], self.num_posterior_samples, axis=0), ms, numpy.sqrt(vs))
        return scipy.special.logsumexp(logps, axis=0) - numpy.log(self.num_posterior_samples)

    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * numpy.random.randn(*ms.shape)
