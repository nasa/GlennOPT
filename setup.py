import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
<<<<<<< HEAD
    name = "glennopt",
    version = "0.0.1",
    author = "Paht Juangphanich",
    author_email = "paht.juangphanich@nasa.gov",
    description = ("Multi-objective optimization tool"),
    license = "Proprietary",
    keywords = "optimization, multi-objective, nsga3",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['base_classes','doe','helpers','nsga3'],
=======
    name = "GlennOPT",
    version = "0.0.1",
    author = "Paht Juangphanich",
    author_email = "paht.juangphanich@nasa.gov",
    description = ("Multi-Objective optimization tool"),
    license = "Proprietary",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['base_classes','doe','helpers','nsga3','test','test_functions'],
>>>>>>> Enhance-Execution
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)