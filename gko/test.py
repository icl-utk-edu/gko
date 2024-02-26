#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import matplotlib
import numpy
import pickle
import scipy
import skopt

from .graph import build_graph
from .mdl4 import M4Model
from .search import SearchPyGMO
from .space import Space


matplotlib.use('Agg')
matplotlib.pyplot.style.use('dark_background')


# problem
kwargs_sample1 = {
    'sample.seed': None,
    'sample.library': 'pydoe',
    'sample.algorithm': 'corr'
}

kwargs_sample2 = {
    'sample.seed': None,
    'sample.library': 'smt',
    'sample.algorithm': 'ese'
}

kwargs_sample3 = {
    'sample.seed': None,
    'sample.library': 'openturns',
    'sample.algorithm': ''
}

kwargs_sample4 = {
    'sample.seed': None,
    'sample.library': 'lhsmdu',
    'sample.algorithm': 'LHS-MDU'
}

# with open('sample_options1.json', 'w') as fp:
#    json.dump(kwargs_sample1, fp)
# with open('sample_options2.json', 'w') as fp:
#    json.dump(kwargs_sample2, fp)
# with open('sample_options3.json', 'w') as fp:
#    json.dump(kwargs_sample3, fp)
# with open('sample_options4.json', 'w') as fp:
#    json.dump(kwargs_sample4, fp)
kwargs = kwargs_sample2.copy()

niter = 5

# problem definition
param1 = skopt.space.Integer(128, 2048, transform="normalize", name="nb")
param23 = skopt.space.Real(0, 1, transform="normalize", name="pq")
space = Space([param1, param23])

samples = {}

# sampling fidelity (0,1) and (0,2)
# [samples[(0,1)], samples[(0,2)]] = sample(
#        27,
#        space,
#        (0,None),
#        model            = None,
#        model_weights    = None,
#        repulsors        = None,
#        repulsors_weight = None,
#        niter            = niter,
#        **kwargs
#        )

# matplotlib.pyplot.clf()
# matplotlib.pyplot.plot(samples[(0,1)][:, 0], samples[(0,1)][:, 1], "o", c='blue')
# matplotlib.pyplot.plot(samples[(0,2)][:, 0], samples[(0,2)][:, 1], "s", c='blue')
# matplotlib.pyplot.savefig('l0f12.png', dpi=300)

# sampling fidelity (0,0)
# samples[(0,0)] = sample(
#        12,
#        space,
#        (0,0),
#        model            = None,
#        model_weights    = None,
#        repulsors        = [samples[(0,1)], samples[(0,2)]],
#        repulsors_weight = [1,1],
#        niter            = niter,
#        **kwargs
#        )
#
# matplotlib.pyplot.plot(samples[(0,0)][:, 0], samples[(0,0)][:, 1], "x", c='red')
# matplotlib.pyplot.savefig('l0f0.png', dpi=300)

# Sampling fidelity (1,1) and (1,2)
# [samples[(1,1)], samples[(1,2)]] = sample(
#        27,
#        space,
#        (1,None),
#        model            = None,
#        model_weights    = None,
#        repulsors        = [samples[(0,0)], samples[(0,1)], samples[(0,2)]],
#        repulsors_weight = [4, 2, 1, 1, 2, 1, 1],
#        niter            = niter,
#        **kwargs
#        )
#
#
# matplotlib.pyplot.clf()
# matplotlib.pyplot.plot(samples[(1,1)][:, 0], samples[(1,1)][:, 1], "o", c='cyan')
# matplotlib.pyplot.plot(samples[(1,2)][:, 0], samples[(1,2)][:, 1], "s", c='cyan')
# matplotlib.pyplot.savefig('l1f12.png', dpi=300)

# Sampling fidelity (1,0)
# samples[(1,0)] = sample(
#        12,
#        space,
#        (1,0),
#        model            = None,
#        model_weights    = None,
#        repulsors        = [samples[(0,0)], samples[(1,1)], samples[(1,2)]],
#        repulsors_weight = [2, 1, 1],
#        niter            = niter,
#        **kwargs
#        )

# matplotlib.pyplot.plot(samples[(1,0)][:, 0], samples[(1,0)][:, 1], "x", c='magenta')
# matplotlib.pyplot.savefig('l1f0.png', dpi=300)

# Save samples
# with open('samples.pkl', 'wb') as fd:
#    pickle.dump(samples, fd)

# Load samples
# with open('samples.pkl', 'rb') as fd:
#    samples = pickle.load(fd)

# Convert samples from normalized space to original space
# for fidelity in samples:
#    print(fidelity, space.inverse_transform(samples[fidelity]))

# Modeling fidelity (0,1)
# def func(x):
#    a = numpy.array(numpy.linalg.norm(x - .5, axis=1)**2, ndmin=2).T
#    return a

# X = {(0,1) : a, (0,2) : b, (0,0) : c}
# X.update({(1,1) : d, (1,2) : e, (1,0) : f})
# Y = {key : func(X[key]) for key in X}


# load data
def load_data():
    with open("data_nb2048_v2/seb_nb2048_v2_normalized.pickl", "rb") as fd:
        Xs = pickle.load(fd)
    X = {(lev, fid): Xs[lev][fidx] for lev in range(4) for fidx, fid in enumerate([1, 2, 0])}

    Y = {}
    for key in X:
        with open("data_nb2048_v2/l%df%d" % (key[0], key[1]), "rb") as fd:
            Y[key] = numpy.array([float(ln) for ln in fd.readlines()], ndmin=2).T

    return (X, Y)


# Xtrain, Ytrain, Xtest, Ytest = load_data()
X, Y = load_data()
Xtest = {key: X[key][20:, :] if key[1] == 0 else X[key] for key in X}
Xtrain = {key: X[key][:20, :] if key[1] == 0 else X[key] for key in X}
Ytest = {key: Y[key][20:, :] if key[1] == 0 else Y[key] for key in Y}
Ytrain = {key: Y[key][:20, :] if key[1] == 0 else Y[key] for key in Y}
# Ytrain = {key: func(Xtrain[key]) for key in Xtrain}
# Ytest = {key: func(Xtest[key]) for key in Xtest}

for key in Xtrain:
    Ytrain[key][:, 0] = (Ytrain[key][:, 0] - min(Ytrain[key][:, 0]))/(max(Ytrain[key][:, 0]) - min(Ytrain[key][:, 0]))
for key in Xtest:
    Ytest[key][:, 0] = (Ytest[key][:, 0] - min(Ytrain[key][:, 0]))/(max(Ytrain[key][:, 0]) - min(Ytrain[key][:, 0]))

# with open("x_normalized.json", "w") as fp:
#    json.dump(encode(Xtrain), fp, indent=2, cls=Encoder)
# with open("y_normalized.json", "w") as fp:
#    json.dump(encode(Ytrain), fp, indent=2, cls=Encoder)

# model
kwargs_model_GP = {
    "model.threads": 6,
    "model.max_iters": 3000,
    "model.verbose": False,
    "model.linear": False,
    "model.nonlinear": True,
    "model.ARD": False,
    "model.restarts": 50,
    "model.type": ['GP', 'GP', 'GP'],
    "model.propagate_var": [True, True, True]
}

kwargs_model_DGP = {
    "model.threads": 40,
    "model.max_iters": 30000,
    "model.verbose": False,
    "model.n_layers": 10,
    "model.type": ['DGP', 'DGP', 'DGP'],
    "model.propagate_var": [False, False, False]
}

# with open("model_options_GP.json", "w'" as fp:
#    json.dump(kwargs_model_GP, fp)
# with open("model_options_DGP.json", "w") as fp:
#    json.dump(kwargs_model_DGP, fp)

kwargs.update(kwargs_model_GP)

# build model
G = build_graph(start_level=0, end_level=3, small_matrix_fidelity=True, partial_facto_fidelity=True, side_links=False)
model = M4Model(X=Xtrain, Y=Ytrain, G=G)

# train model
model.train(**kwargs)

model.dump("model.pickl")
# model = M4Model.load("model.pkl")

# search
kwargs_search = {
    "search.minimize": False,  # minimize or Maximize the objective function
    "search.threads": 40,  # number of threads in each thread group handling one task
    "search.algo": "cmaes",
    "search.udi": "thread_island",
    "search.bfe": "bfe",
    "search.pop_": 1000,  # population size in pgymo
    "search.gen": 100,   # number of evolution generations in pgymo
    "search.evolve": 10,   # number of times migration in pgymo
    "search.max_iters": 10,   # max number of searches to get results respecting the constraints
    "search.batch": True,  # batch evaluations in search or use of archipelagio
}

# with open('dearch_options.json', 'w') as fp:
#    json.dump(kwargs_search, fp)

kwargs.update(kwargs_search)
searcher = SearchPyGMO()

candidates = dict()
for key in model.M:
    candidates[key] = searcher.search(1, space, model, fidelity=key, **kwargs)


def impplot(x, y, z, key, candidates=None, name=''):
    "implementation of local plot"
    matplotlib.pyplot.clf()
#   matplotlib.pyplot.contour(x, y, z, 100)
#   matplotlib.pyplot.pcolor(x, y, z, cmap=cm.plasma, linewidth=0, antialiased=True)
    matplotlib.pyplot.imshow(z,
            cmap=matplotlib.cm.plasma,
            # vmin=y[:,0].min(),
            # vmax=y[:,0].max(),
            vmin=z.min(),
            vmax=z.max(),
            origin="lower",
            extent=[x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max()])
    matplotlib.pyplot.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap=matplotlib.cm.plasma)
    if (candidates is not None):
        matplotlib.pyplot.scatter(candidates[:, 0], candidates[:, 1])
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%d_%d_%s.png' % (key[0], key[1], name), dpi=600)


muvar = dict()
for key in model.M:
    print('MSE', numpy.linalg.norm(Ytest[key] - model.predict(Xtest[key], key, **kwargs)[key][0]))

    x = numpy.arange(0, 100, 1) / 100.0
    y = numpy.arange(0, 100, 1) / 100.0
    xx, yy = numpy.meshgrid(x, y)
    xy = numpy.concatenate((xx.flatten()[:, None], yy.flatten()[:, None]), axis=1)

    muvar = model.predict(xy, key, muvar, **kwargs)
    mu, var = muvar[key]
    mu = mu.reshape(xx.shape)
    var = var.reshape(xx.shape)
    impplot(Xtrain[key], Ytrain[key], mu, key, candidates[key], 'mu')
    impplot(Xtrain[key], Ytrain[key], var, key, candidates[key], 'var')

    z = scipy.interpolate.Rbf(Xtrain[key][:, 0], Xtrain[key][:, 1], Ytrain[key][:, 0], function='gaussian')(xx, yy)
    impplot(Xtrain[key], Ytrain[key], z, key, candidates[key], 'train')

    z = scipy.interpolate.Rbf(X[key][:, 0], X[key][:, 1], Y[key][:, 0], function='gaussian')(xx, yy)
    impplot(X[key], Y[key], z, key, candidates[key], 'ground')
