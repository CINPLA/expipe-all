import click
from expipecli.utils.misc import lazy_import

@lazy_import
def expipe():
    import expipe
    return expipe

@lazy_import
def warnings():
    import warnings
    def _warning(
        message,
        category = UserWarning,
        filename = '',
        lineno = -1):
        print()
        print('WARNING:', message)
        print()

    warnings.showwarning = _warning
    return warnings

@lazy_import
def PAR():
    from expipe_plugin_cinpla.tools.config import load_parameters
    return load_parameters()


def require_project(): # TODO not needed bla
    from expipecli.main import load_config
    config = load_config()
    if config['local_root'] is None:
        print('Unable to locate expipe configurations.')
        return None
    assert config['local']['type'] == 'project'
    server = expipe.require_project(path=config['local_root'], name=config['local_root'].stem)
    return server


def get_project():
    from expipecli.main import load_config
    config = load_config()
    if config['local_root'] is None:
        print('Unable to locate expipe configurations.')
        return None
    assert config['local']['type'] == 'project'
    server = expipe.get_project(path=config['local_root'], name=config['local_root'].stem)
    return server

@lazy_import
def pd():
    import pandas as pd
    return pd

@lazy_import
def sig_tools():
    import exana.misc.signal_tools as sig_tools
    return sig_tools

@lazy_import
def pyopenephys():
    import pyopenephys
    return pyopenephys

@lazy_import
def openephys():
    from expipe_io_neuro import openephys
    return openephys

@lazy_import
def pyxona():
    import pyxona
    return pyxona

@lazy_import
def platform():
    import platform
    return platform

@lazy_import
def csv():
    import csv
    return csv

@lazy_import
def json():
    import json
    return json

@lazy_import
def axona():
    from expipe_io_neuro import axona
    return axona

@lazy_import
def os():
    import os
    return os

@lazy_import
def shutil():
    import shutil
    return shutil

@lazy_import
def datetime():
    from datetime import datetime
    return datetime

@lazy_import
def timedelta():
    from datetime import timedelta
    return timedelta

@lazy_import
def subprocess():
    import subprocess
    return subprocess

@lazy_import
def tarfile():
    import tarfile
    return tarfile

@lazy_import
def paramiko():
    import paramiko
    return paramiko

@lazy_import
def getpass():
    import getpass
    return getpass

@lazy_import
def tqdm():
    from tqdm import tqdm
    return tqdm

@lazy_import
def scp():
    import scp
    return scp

@lazy_import
def neo():
    import neo
    return neo

@lazy_import
def exdir():
    import exdir
    return exdir

@lazy_import
def pq():
    import quantities as pq
    return pq

@lazy_import
def logging():
    import logging
    return logging

@lazy_import
def np():
    import numpy as np
    return np

@lazy_import
def copy():
    import copy
    return copy

@lazy_import
def scipy():
    import scipy
    import scipy.io
    return scipy

@lazy_import
def glob():
    import glob
    return glob

@lazy_import
def el():
    import elephant as el
    return el

@lazy_import
def sys():
    import sys
    return sys

@lazy_import
def pprint():
    import pprint
    return pprint

@lazy_import
def collections():
    import collections
    return collections
