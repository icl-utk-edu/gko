#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json

import numpy
import pygmo
import scipy

from .jsonutil import Encoder, encode
from .mdl4 import M4Model


class UDP:
    def __init__(self, search_space, fidelity, **kwargs):
        self.search_space = search_space
        self.bounds = list(zip(*[dim.transformed_bounds for dim in search_space]))
        self.fidelity = fidelity
        self.kwargs = kwargs

    def get_bounds(self) -> list:
        return self.bounds

    def get_nobj(self) -> int:
        return 1

    def has_batch_fitness(self) -> bool:
        return True

    def fitness(self, dv):
        return self.batch_fitness(dv)

    def batch_fitness(self, dvs):
        n = len(self.search_space)
        m = len(dvs) // n
        nf = 1
        dvsNorm = dvs.reshape((m, n))
        points = self.search_space.inverse_transform(dvsNorm)

        res = numpy.full((m, nf), numpy.inf)

        # check if points are already in the database
        # conds = numpy.array([(self.modeler.X[self.fidelity] == x).all(axis=1).any() for x in points])
        # idxs = numpy.where(conds)[0]
        # if (len(idxs) > 0):
        #     Compute acquisition of valid points
        res = numpy.array(self.batch_ei(dvsNorm))  # [idxs])

        # batch_fitness must return a 1D array
        res = res.flatten()

        return res

    def batch_ei(self, x):

        """ Expected Improvement """

        sign = 1.0 if self.kwargs['search.minimize'] else -1.0
        x = numpy.array(x, ndmin=2)
        ymin = sign * self.modeler.Y[self.fidelity].max()

        EI = []
        (muss, varss) = self.modeler.predict(x, fidelity=self.fidelity, **self.kwargs)[self.fidelity]
        for i in range(len(x)):
            mu = sign * muss[i][0]
            var = max(1e-18, varss[i][0])
            std = numpy.sqrt(var)
            chi = (ymin - mu) / std
            Phi = 0.5 * (1.0 + scipy.special.erf(chi / numpy.sqrt(2)))
            phi = numpy.exp(-0.5 * chi**2) / numpy.sqrt(2 * numpy.pi * var)
            EI.append(-((ymin - mu) * Phi + var * phi))

        return EI


class SearchPyGMO:
    def search(self, n_sol, search_space, model, fidelity, **kwargs):
        self.modeler = model

        # prob
        udp = UDP(search_space, fidelity, **kwargs)
        prob = pygmo.problem(udp)

        # algo

        try:
            uda = getattr(pygmo, kwargs["search.algo"])(gen=kwargs["search.gen"])
            algo = pygmo.algorithm(uda)
        except AttributeError:
            raise ValueError("Unknown optimization algorithm " + kwargs["search.algo"])

        # island
        try:
            udi = getattr(pygmo, kwargs["search.udi"])()
        except AttributeError:
            raise ValueError("Unknown user-defined-island " + kwargs["search.udi"])

        # batch fitness evaluator
        if kwargs['search.bfe'] is None:
            bfe = None
        else:
            try:
                bfe = getattr(pygmo, kwargs["search.bfe"])()
            except AttributeError:
                raise ValueError("Unknown batch fitness evaluator " + kwargs["search.bfe"])

#        uda.set_bfe(bfe)

        solutions = []
        cond = False
        cpt = 0
        while ((not cond) and (cpt < kwargs['search.max_iters'])):

            # pop

            # if (kwargs['search.batch']):
            pop = pygmo.population(prob, size=kwargs['search.pop_size'],
                                   b=bfe, seed=cpt)
            #     pop = algo.evolve(pop, n = kwargs['search.evolve'])
            #     champions_f = pop.get_f()
            #     champions_x = pop.get_x()
            # else:

            archi = pygmo.archipelago(
                    n=kwargs['search.threads'],
                    t=pygmo.topology(udt=pygmo.unconnected()),
                    algo=algo,
                    pop=pop,
                    udi=udi,
                    )
            # prob = prob, pop_size = kwargs['search.pop_size'], b = bfe

            # search

            archi.evolve(n=kwargs['search.evolve'])
            archi.wait()

            champions_f = archi.get_champions_f()
            champions_x = archi.get_champions_x()
            indexes = list(range(len(champions_f)))
            indexes.sort(key=champions_f.__getitem__)
            for idx in indexes:
                if champions_f[idx] < numpy.inf:
                    solutions.append(numpy.array(champions_x[idx]))
                    if len(solutions) >= n_sol:
                        cond = True
                        break

            cpt += 1

        solutions = numpy.array(solutions)

        return solutions


def search(args) -> None:
    options = json.load(args.model_opts)
    options.update(json.load(args.search_opts))

    lvl = args.level[0]
    fid = args.fidelity

    model = M4Model.load(args.model.name)

    searcher = SearchPyGMO()

    candidates = {(lvl, fid): searcher.search(1, space[lvl], model,
                                              fidelity=(lvl, fid), **options)}

    json.dump(encode(candidates), args.outfile, indent=2, cls=Encoder)
