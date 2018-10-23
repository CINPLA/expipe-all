import click
from expipecli.utils.misc import lazy_import


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
def expipe():
    import expipe
    return expipe

@lazy_import
def warnings():
    import warnings
    return warnings

@lazy_import
def PAR():
    from expipe_plugin_cinpla.tools.config import load_parameters
    return load_parameters()

@lazy_import
def yaml():
    import yaml
    return yaml

@lazy_import
def pprint():
    import pprint
    return pprint

@lazy_import
def imp():
    import imp
    return imp

@lazy_import
def collections():
    import collections
    return collections

@lazy_import
def plt():
    import matplotlib.pyplot as plt
    return plt
