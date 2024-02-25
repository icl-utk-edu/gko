#! /usr/bin/env python3
# -*- coding: ascii -*-
# vim: set fileencoding=ascii


__all__ = []
__author__ = "Piotr Luszczek"
__version__ = "23.12"


import logging as _gko_logging
import tempfile as _gko_tempfile


_LogFd, _LogName = _gko_tempfile.mkstemp(prefix="gko", suffix=".log")
_gko_logging.basicConfig(
    filename=_LogName,
    level=_gko_logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
)
