#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json

import numpy

from .mdl4 import M4Model


def predict(args):
    options = json.load(args.options)

    model = M4Model.load(args.model.name)

    param1 = args.param1
    param23 = args.param2, args.param3
    param4 = args.param4

    pz = [param1, param23] if param4 is None else [param1, param23, param4]
    X = space[args.level[0]].transform(numpy.array([pz], dtype=object))

    muvar = model.predict(X, (args.level[0], args.fidelity), {}, **options)

    # y_min = min( model.Y[(args.level[0], args.fidelity)] )
    # y_max = max( model.Y[(args.level[0], args.fidelity)] )

    # mu = y_min + (y_max - y_min) * muvar[(args.level[0], args.fidelity)][0]
    # var = (y_max - y_min) * muvar[(args.level[0], args.fidelity)][1]
    # muvar[(args.level[0], args.fidelity)] = (mu, var)

    print("[%d:%d] Param(%d, %s, %s)\tPrediction %f Gflops/s\tConfidence %f" %
          (args.level[0], args.fidelity, param1, param23, param4,
           muvar[(args.level[0], args.fidelity)][0],
           muvar[(args.level[0], args.fidelity)][1]))
