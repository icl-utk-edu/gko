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
    "f": {
        "lopt": "--fidelity",
        "kw": {
            "help": "fidelity to use for the command",
            "type": int,
        },
    },
    "g": {
        "lopt": "--graph",
        "kw": {
            "help": "fidelity to use for the command",
            "type": argparse.FileType("r"),
        },
    },
    "i": {
        "lopt": "--infile",
        "kw": {
            "help": "filename for the command's input",
            "type": argparse.FileType("r"),
        },
    },
    "l": {
        "lopt": "--level",
        "kw": {
            "help": "filename for the command's graph",
            "type": int,
        },
    },
    "n": {
        "lopt": "--samples",
        "kw": {
            "help": "number of samples",
            "type": int,
        },
    },
    "O": {
        "lopt": "--options",
        "kw": {
            "help": "filename for the addition options for the command",
            "type": argparse.FileType("r"),
        },
    },
    "o": {
        "lopt": "--outfile",
        "kw": {
            "help": "filename for the command's output",
            "type": argparse.FileType("w"),
        },
    },
}


def parse(argv):
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
         ("convert", "io", "Convert in/out files between origin and noermalized data spaces"),
         ("sample", "fgilnOo", "Sample the input space"),
         ):
        parsers[cmd] = command.add_parser(cmd, help=hlp)
        for f in flgs:
            parsers[cmd].add_argument(_CommonOpts[f]["lopt"],
                **_CommonOpts[f]["kw"])

    parsers["convert"].add_argument("-d", "--direction",
        choices=("orig2norm", "norm2orig"))

    return main.parse_args(args=argv[1:])


def main(argv):
    args = parse(argv)
    if args.command is None:
        print("GKO or Generative Kernel Optimization.\n\n"
              "Please specify one of the available commands or "
              "use '-h' flag for more information.")
        return 127

    return dispatch(args)
