#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


"""
Main entry point into GKO or Generative Kernel Optimization.

@author Piotr Luszczek
"""


import sys


if sys.version_info < (3,):
    raise RuntimeError("Only Python 3+ supported")


# this doesn't parse with Python 2.7: let's count __main__.py isn't loaded
from .argvparse import main


if "__main__" == __name__:
    sys.exit(main(sys.argv))
