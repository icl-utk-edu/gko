#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
from .jsonutil import Encoder, decode, encode


def convert(args) -> None:
    if "orig2norm" == args.direction:
        fcn = space[key[0]].transform
    else:
        fcn = space[key[0]].invtransform
    out = {key: fcn(val)
           for key, val in decode(json.load(args.infile)).items()}
    json.dump(encode(out), args.outfile, indent=2, cls=Encoder)
