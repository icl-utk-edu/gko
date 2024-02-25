#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


"""
Main entry point into GKO or Generative Kernel Optimization and command
line argument parsing.

@author Piotr Luszczek
"""


import argparse
from .commands import dispatch


_CommonOpts = {
    "1": {
        "l": "--parameter",
        "k": {
            "help": "parameter number start at 1 that is predicted by the command'",
            "choices": (1,),  # this will be replaced by arg parsing below
        },
    },
    "c": {
        "l": "--candidates",
        "k": {
            "help": "filename for the samples updated by the command'",
            "type": argparse.FileType("r"),
        },
    },
    "f": {
        "l": "--fidelity",
        "k": {
            "help": "fidelity to use for the command",
            "type": int,
            "nargs": "+",
        },
    },
    "g": {
        "l": "--graph",
        "k": {
            "help": "fidelity to use for the command",
            "type": argparse.FileType("r"),
        },
    },
    "i": {
        "l": "--infile",
        "k": {
            "help": "filename for the command's input",
            "type": argparse.FileType("r"),
        },
    },
    "l": {
        "l": "--level",
        "k": {
            "help": "filename for the command's graph",
            "type": int,
        },
    },
    "M": {
        "l": "--model-opts",
        "k": {
            "help": "filename with the model options for the command",
            "type": argparse.FileType("r"),
        },
    },
    "m": {
        "l": "--model",
        "k": {
            "help": "filename with the model for the command",
            "type": argparse.FileType("r"),
        },
    },
    "n": {
        "l": "--sample-count",
        "k": {
            "help": "number of samples",
            "type": int,
        },
    },
    "O": {
        "l": "--options",
        "k": {
            "help": "filename for the addition options for the command",
            "type": argparse.FileType("r"),
        },
    },
    "o": {
        "l": "--outfile",
        "k": {
            "help": "filename for the command's output",
            "type": argparse.FileType("w"),
        },
    },
    "P": {
        "l": "--perf-candid",
        "k": {
            "help": "filename for the performance candidates for the command'",
            "type": argparse.FileType("r"),
        },
    },
    "p": {
        "l": "--perf",
        "k": {
            "help": "filename for the performance values for the command'",
            "type": argparse.FileType("r"),
        },
    },
    "S": {
        "l": "--search-opts",
        "k": {
            "help": "filename with the search options for the command",
            "type": argparse.FileType("r"),
        },
    },
    "s": {
        "l": "--samples",
        "k": {
            "help": "filename for the samples updated by the command'",
            "type": argparse.FileType("r"),
        },
    },
}


def parse(argv: [str]):
    main = argparse.ArgumentParser(
        prog=argv[0],
        description="GKO or Generative Kernel Optimization",
        epilog="The project is currently under development and new commands are possible in the future.",
        prefix_chars="-",
        usage="%(prog)s <command> [<command>] [options]")

    main.add_argument("-c", "--config", type=argparse.FileType("r"),
        help="path to main configuration file")

    command = main.add_subparsers(
        title="Actions",
        description="Choose one of possible commands and their arguments.",
        help="names for specific commands",
        dest="command")

    parsers = dict()
    for cmd, flgs, hlp in (
         ("convert", "io", "convert in/out files between origin and noermalized data spaces"),
         ("merge", "cflsPp", "sample the input space"),
         ("model", "fgilOops", "model the input space with a learned model"),
         ("plot", "cflMmops", "plot the samples and predictions"),
         ("predict", "flmO4", "predict performance values in output space"),
         ("search", "flMmSo", "search optimal values in the output space"),
         ("sample", "fgilnOo", "sample the input space"),
         ):
        parsers[cmd] = command.add_parser(cmd, help=hlp)
        for f in flgs:
            v = _CommonOpts["1" if f.isdigit() else f]
            if f.isdigit():
                v["k"]["choices"] = list(range(int(f)))
            parsers[cmd].add_argument(v["l"], **v["k"])

    parsers["convert"].add_argument("-d", "--direction",
        choices=("orig2norm", "norm2orig"))

    return main.parse_args(args=argv[1:])


def main(argv: [str]):
    args = parse(argv)
    if args.command is None:
        print("GKO or Generative Kernel Optimization.\n\n"
              "Please specify one of the available commands or "
              "use '-h' flag for more information.")
        return 127

    return dispatch(args)
