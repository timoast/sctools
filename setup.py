#!/usr/bin/env python

from setuptools import setup
import versioneer
import subprocess


def samtools():
    v = subprocess.check_output(['samtools', '--version']).split()[1].split('.')
    major = int(v[0])
    minor = int(v[1])
    if major > 0.8:
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
    scripts = ["sctools/sctools.py"],
    author_email = 'timstuart90@gmail.com',
    url = 'https://github.com/timoast/sctools',
    packages = ['sctools'],
    test_suite="nose.collector"
)
