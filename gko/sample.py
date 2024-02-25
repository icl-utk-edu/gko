#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


import json
import os
import sys
from .space import get_levels
from .jsonutil import Encoder, encode


def sample(args) -> None:
    print(args)
    levels = get_levels()
    lvl = args.level[0]

    sample_counts = args.sample_count if len(args.sample_count) > 1 \
        else 2 * [args.sample_count[0]]

    try:
        if os.path.samefile(args.infile.name, args.outfile.name):
            raise ValueError("Input and output files must be different.")
    except TypeError:  # thrown if filenames are None
        pass

    try:
        samples = dict() if args.infile is None else json.load(args.infile)

    except json.decoder.JSONDecodeError:
        print("Input file had JSON decoding issue(s).", file=sys.stderr)
        raise

    if not samples and not args.fidelity:
        raise FileNotFoundError("Input file required.")

    if not args.options:
        raise ValueError("Options file required.")

    try:
        options = json.load(args.options)

    except TypeError:
        print("Missing options file.", file=sys.stderr)
        raise

    except json.decoder.JSONDecodeError:
        print("Options file had JSON decoding issue(s).", file=sys.stderr)
        raise

    if 0 == lvl:
        if args.fidelity == 12:
            [samples[(lvl, 1)], samples[(lvl, 2)]] = sample(
                    sample_counts,
                    levels[lvl],
                    (lvl, None),
                    model=None,
                    model_weights=None,
                    repulsors=None,
                    repulsors_weight=None,
                    niter=options['sample.niter'],
                    **options
                    )
        else:
            samples[(lvl, args.fidelity)] = sample(
                    sample_counts,
                    levels[lvl],
                    (lvl, args.fidelity),
                    model=None,
                    model_weights=None,
                    repulsors=[samples[(lvl, 1)], samples[(lvl, 2)]],
                    repulsors_weight=[1, 1],
                    niter=options['sample.niter'],
                    **options
                    )
    else:
        if args.fidelity == 12:
            [samples[(lvl, 1)], samples[(lvl, 2)]] = sample(
                    sample_counts,
                    levels[lvl],
                    (lvl, None),
                    model=None,
                    model_weights=None,
                    repulsors=[samples[(lvl-1, 0)], samples[(lvl-1, 1)],
                               samples[(lvl-1, 2)]],
                    repulsors_weight=[4, 2, 1, 1, 2, 1, 1],
                    niter=options['sample.niter'],
                    **options
                    )
        else:
            samples[(lvl, args.fidelity)] = sample(
                    sample_counts,
                    levels[lvl],
                    (lvl, args.fidelity),
                    model=None,
                    model_weights=None,
                    repulsors=[samples[(lvl-1, 0)], samples[(lvl, 1)],
                               samples[(lvl, 2)]],
                    repulsors_weight=[2, 1, 1],
                    niter=options['sample.niter'],
                    **options
                    )

    json.dump(encode(samples), args.outfile, indent=2, cls=Encoder)
