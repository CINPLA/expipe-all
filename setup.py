# -*- coding: utf-8 -*-
import sys
import setuptools
import subprocess
from setuptools import setup, Command
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop


def setupInstall(cmd, arg):
    dep_list = [
        'python-neo',
        'exdir',
        'expipe',
        'expipe-plugin-cinpla',
        'expipe-cli',
        'exdir-cli',
        'expipe-io-neuro',
        'exana',
        'pyxona',
        'py-open-ephys'
    ]

    browser_list = [
        'expipe-browser',
        'exdir-browser'
    ]

    phy_list = [
        'phy',
        'phy-contrib'
    ]
    if arg == 'browser':
        print('Installing expipe with browsers')
        dep_list += browser_list
    elif arg == 'phy':
        print('Installing expipe with phy')
        dep_list += phy_list
    elif arg == 'all':
        print('Installing expipe with phy and browser')
        dep_list += phy_list
        dep_list += browser_list
    else:
        print('Installing expipe without browsers')
    for dep in dep_list:
        subprocess.call([sys.executable, "setup.py", cmd], cwd=dep)


class CustomInstall(Command):
    user_options = [
        ('extra=', None, 'Specify extras to install.'),
    ]

    def initialize_options(self):
        self.extra = None

    def finalize_options(self):
        assert self.extra in (None, 'browser', 'phy', 'all'), 'Invalid extra!'

    def run(self):
        if 'develop' in sys.argv:
            setupInstall('develop', self.extra)
            # _develop.run(self)
        elif 'install' in sys.argv:
            setupInstall('install', self.extra)
            # _install.run(self)
        elif 'uninstall' in sys.argv:
            subprocess.call("./uninstall.sh", shell=True)


cmdclass = {'install': CustomInstall,
            'develop': CustomInstall,
            'uninstall': CustomInstall}

long_description = open("README.md").read()

setup(name="expipe-dev",
      version='0.1',
      entry_points=None,
      author="CINPLA",
      author_email="",
      maintainer="Mikkel Elle Lepperød",
      maintainer_email="m.e.lepperod@medisin.uio.no",
      long_description=long_description,
      url="https://github.com/CINPLA/expipe-dev",
      platforms=['Linux', "Windows"],
      cmdclass=cmdclass,
      )
