#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
import numpy


class Encoder(json.JSONEncoder):
    def default(self, o):
        for t, f in (
            (numpy.floating, float),
            (numpy.integer, int),
            (numpy.ndarray, lambda x: x.tolist()),
        ):
            if isinstance(o, t):
                return f(o)
        return super(Encoder, self).default(o)


def encode(mapping):
    return {str(k): v for k, v in mapping.items()}


def decode(mapping):
    return {eval(k): v for k, v in mapping.items()}
