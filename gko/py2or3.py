#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


"""
Main entry point into GKO or Generative Kernel Optimization.

@author Piotr Luszczek
"""


import sys


if sys.version_info < (3,):  # detect wrong Python early with version 2/3 syntax
    sys.stdout.write("Only Python 3+ supported.\n")
    sys.exit(127)


import gko.argvparse
sys.exit(gko.argvparse.main(sys.argv))
