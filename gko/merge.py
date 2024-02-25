#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
import numpy
from .jsonutil import decode, Encoder, encode


def merge(args) -> None:
    key = args.level[0], args.fidelity

    # merge samples and their candidates
    if args.candidates is not None:
        candidates = decode(json.load(args.candidates))
        value = candidates[key]

        with open(args.samples, "r") as fd:
            newcand = decode(json.load(fd))

        newcand[key] = numpy.concatenate((numpy.array(newcand[key]), value))

        with open(args.samples, "w") as fd:
            json.dump(encode(newcand), fd, indent=2, cls=Encoder)

    # merge perf values and their candidates
    if args.perf_candid:
        try:
            with open(args.perf, "r") as fd:
                newperf = decode(json.load(fd))
        except Exception:  # nothing to merge because no file, or JSON issue
            newperf = {}
        perfcand = args.perf_candid

        perfcand = numpy.array(perfcand, ndmin=2).reshape((len(perfcand), 1))
        if key in newperf.keys():
            newperf[key] = numpy.concatenate((
                numpy.array(newperf[key]), perfcand))
        else:
            newperf[key] = perfcand

        with open(args.perf, "w") as fd:
            json.dump(encode(newperf), fd, indent=2, cls=Encoder)
