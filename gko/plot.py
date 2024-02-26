#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
import matplotlib
import numpy

from .jsonutil import decode
from .mdl4 import M4Model

matplotlib.use('Agg')
matplotlib.pyplot.style.use('dark_background')


def plot(args):
    def convGrid(grid_str):
        return float(grid_str.split(",")[0])

    def impplot(x=None, y=None, z=None, candidates=None, name=""):
        matplotlib.pyplot.clf()
        # matplotlib.pyplot.contour(x, y, z, 100)
        # matplotlib.pyplot.pcolor(x, y, z, cmap=matplotlib.cm.plasma, linewidth=0, antialiased=True)
        if z is not None:
            matplotlib.pyplot.imshow(z, cmap=matplotlib.cm.plasma, vmin=z.min(), vmax=z.max(), origin='lower', extent=[0., 1., 0., 1.])  # vmin=y[:,0].min(), vmax=y[:,0].max(),
            # matplotlib.pyplot.colorbar(cax=matplotlib.pyplot.gca())
            matplotlib.pyplot.colorbar()
        if x is not None:
            xDataOriginal = False
            # detect whether the data are original or normalized
            if isinstance(x[0, 1], str) and len(x[0, 1].split(',')) == 2:
                xDataOriginal = True
                convertGrid = {key: convGrid(key) for key in x[:, 1]}
                gridLabel = [k for k, _ in sorted(convertGrid.items(), key=lambda item: item[1])]
                x[:, 1] = [convertGrid[grid_val] for grid_val in x[:, 1]]

            if y is not None:
                matplotlib.pyplot.scatter(x[:, 0].astype(numpy.float), x[:, 1].astype(numpy.float), c=y[:, 0].astype(numpy.float), cmap=matplotlib.cm.plasma)
                # matplotlib.pyplot.scatter(x[:,0].astype(numpy.float), x[:,1].astype(numpy.float), color='black')
                # matplotlib.pyplot.scatter( [0.5], [0.7], color='black')
            else:
                matplotlib.pyplot.scatter(x[:, 0].astype(numpy.float), x[:, 1].astype(numpy.float), cmap=matplotlib.cm.plasma)
            if xDataOriginal:
                matplotlib.pyplot.yticks([convertGrid[v] for v in gridLabel], gridLabel)
        if candidates is not None:
            matplotlib.pyplot.scatter(candidates[:, 0], candidates[:, 1], marker='x', c='green')
        matplotlib.pyplot.grid()

        matplotlib.pyplot.savefig(name, dpi=600)

    lvl = args.level[0]
    fid = args.fidelity
    key = lvl, fid

    mu = None
    var = None
    X = None
    Y = None
    C = None
    Ysub = None
    y_min = 0.0
    y_max = 1.0

    # plot the model
    if args.samples is not None:
        X = numpy.array(decode(json.load(args.samples))[key])

        if args.perf is not None:
            Y = numpy.array(decode(json.load(args.perf))[key])
            y_min = min(Y)
            y_max = max(Y)

    if args.candidates is not None:
        C = numpy.array(decode(json.load(args.candidates))[key])

    if args.model is not None:
        if args.options is None:
            raise ValueError("Argument 'options' reauired")

        options = json.load(args.options)
        model = M4Model.load(args.model.name)

    for param4 in space[lvl][2].categories:
        if args.model is not None:
            x = numpy.arange(0, 100, 1) / 100.0
            y = numpy.arange(0, 100, 1) / 100.0
            z = numpy.array(space[lvl][2].transform([param4]))

            # x = numpy.arange(0, 10, 1)/10.
            # y = numpy.arange(0, 10, 1)/10.
            xx, yy, zz = numpy.meshgrid(x, y, z)
            xy = numpy.concatenate((xx.flatten()[:, None], yy.flatten()[:, None], zz.flatten()[:, None]), axis=1)

            muvar = model.predict(xy, key, {}, **options)
            mu, var = muvar[key]
            # mu = y_min + (y_max - y_min) * mu
            # var = (y_max - y_min) * var
            mu = mu.reshape(xx.shape)
            var = var.reshape(xx.shape)

        # Filter the X and Y based on the type of param4
        idx = numpy.where(space[lvl][2].inverse_transform(X[:, 2]) == param4)
        Xsub = X[idx]

        if args.perf is not None:
            Ysub = Y[idx]

        if mu is None:
            impplot(Xsub, Ysub, mu, C, "%s_%s_l%d_f%d.png" % (args.outfile, param4, lvl, fid))
        else:
            impplot(Xsub, Ysub, mu, C, "%s_%s_l%d_f%d_mu.png" % (args.outfile, param4, lvl, fid))
            impplot(Xsub, Ysub, var, C, "%s_%s_l%d_f%d_var.png" % (args.outfile, param4, lvl, fid))
