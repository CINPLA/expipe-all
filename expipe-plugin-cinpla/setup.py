# -*- coding: utf-8 -*-
from setuptools import setup

from setuptools import setup, find_packages
import versioneer

long_description = open("README.md").read()

setup(
    name="expipe-plugin-cinpla",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    author="CINPLA",
    author_email="",
    maintainer="Mikkel Elle Lepperød",
    maintainer_email="m.e.lepperod@medisin.uio.no",
    platforms=['Linux', "Windows"],
    description="Plugins for the CINPLA lab",
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'plugin-expipe-cinpla = expipe_plugin_cinpla.cli.main:reveal'
        ],
    },
)
