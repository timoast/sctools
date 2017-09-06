#!/usr/bin/env python

from setuptools import setup
import versioneer
import subprocess
import sys


def samtools():
    if (sys.version_info > (3, 0)):
        v = subprocess.check_output(['samtools', '--version']).decode().split()[1].split('.')
    else:
        v = subprocess.check_output(['samtools', '--version']).split()[1].split('.')
    major = int(v[0])
    minor = int(v[1])
    if major >= 1:
        return True
    return False


if __name__ == "__main__":
    if not samtools():
        raise Exception("sctools requires samtools >= v1")


setup(
    name = 'sctools',
    version = versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="sctools: tools for single cell RNA-seq analysis",
    author = 'Tim Stuart',
    install_requires = [
        'pysam>0.8',
    ],
    scripts = ["scripts/sctools"],
    author_email = 'timstuart90@gmail.com',
    url = 'https://github.com/timoast/sctools',
    packages = ['sctools'],
    test_suite="nose.collector"
)
