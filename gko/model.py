#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
import networkx
import numpy
from .graph import build_graph
from .jsonutil import decode
from .mdl4 import M4Model


def model(args) -> None:
    # load the data
    X = {k: numpy.array(v)
         for k, v in decode(json.load(args.samples)).items()}
    Y = {k: numpy.array(v).astype(numpy.float)
         for k, v in decode(json.load(args.perf)).items()}
    options = json.load(args.options)

    # get the graph
    if args.graph is not None:
        G = json.load(args.graph)

    else:
        # end_lvl = max([key for key in space])
        end_lvl = max([key[0] for key in X])

        # if lvl == 0 and args.fidelity == 1
        # if lvl == 0 and (args.fidelity == 1 or args.fidelity == 2):
        if 0 == end_lvl and (0, 0) not in Y:
            # create special graph for now
            G = networkx.DiGraph()
            if (0, 1) in Y:
                G.add_node((0, 1))
            if (0, 2) in Y:
                G.add_node((0, 2))

        else:
            G = build_graph(
                start_level=0, end_level=end_lvl,
                small_matrix_fidelity=options['model.small_matrix_fidelity'],
                partial_facto_fidelity=options['model.partial_facto_fidelity'],
                side_links=options['model.side_links']
                )

    # assuming x is normalized and so is Y no need normalize_inplace()
    model = M4Model(X=X, Y=Y, G=G)
    model.train(**options)
    model.dump(args.outfile.name)
