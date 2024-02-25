#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


"""
GKO commands

@author Piotr Luszczek
"""


import gko
# these make command sub-modules available
from .sample import sample as _


def dispatch(args) -> int:
    try:
        cmd = getattr(getattr(gko, args.command), args.command)

    except AttributeError:
        print("Unknown command '{}'".format(args.command))
        return 127

    cmd(args)

    return 0
