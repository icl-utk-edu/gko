#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


"""
GKO commands

@author Piotr Luszczek
"""


import gko
# these make command sub-modules available
from .convert import convert
from .merge import merge
from .model import model
from .predict import predict
from .sample import sample
from .search import search


def dispatch(args) -> int:
    try:
        cmd = getattr(getattr(gko, args.command), args.command)

    except AttributeError:
        print("Unknown command '{}'".format(args.command))
        return 127

    cmd(args)

    return 0
