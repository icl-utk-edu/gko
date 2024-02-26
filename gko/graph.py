#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import networkx


def build_graph(start_level=0, end_level=1, small_matrix_fidelity=True,
                partial_facto_fidelity=True, side_links=True) \
                -> networkx.DiGraph:

    G = networkx.DiGraph()
    for lev in range(start_level, end_level + 1):
        G.add_node((lev, 0))
        if lev > start_level:
            G.add_edge((lev - 1, 0), (lev, 0))
        if small_matrix_fidelity:
            G.add_node((lev, 1))
            G.add_edge((lev, 1), (lev, 0))
            if side_links and lev > start_level:
                G.add_edge((lev - 1, 1), (lev, 1))
        if partial_facto_fidelity:
            G.add_node((lev, 2))
            G.add_edge((lev, 2), (lev, 0))
            if side_links and (lev > start_level):
                G.add_edge((lev - 1, 2), (lev, 2))

    return G
