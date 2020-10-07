#!/usr/bin/env python
import os
from setuptools import setup,find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

setup(
    name = "GlennOPT",
    version = "1.0.1",
    author = "Paht Juangphanich",
    author_email = "paht.juangphanich@nasa.gov",
    description = ("Multi and single objective optimization tool for cfd applications."),
    license = "License.md",
    keywords = "multi-objective, optimization, nsga, differential evolution",
    url = "https://github.com/nasa/GlennOPT",
    packages=['base_classes',
    package_dir={'': 'glennopt'},  # Optional

)