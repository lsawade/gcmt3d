""" :noindex:
Setup.py file with generic info
"""
import os
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as TestCommand

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    """From Wenjie Lei 2019"""
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname


long_description = "%s" % read("README.md")

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name="GCMT3D",
    description=("Global 3D Centroid Moment Tensor Inversion"),
    long_description=long_description,
    version="0.0.1",
    author="Lucas Sawade",
    author_email="lsawade@princeton.edu",
    license='GNU Lesser General Public License, Version 3',
    keywords="Global CMT, Inversion, Moment Tensor",
    url='https://github.com/lsawade/GCMT3D',
    package_dir={"": "src"},
    packages=find_packages(),
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: Alpha",
        "Topic :: Utilities",
        "License :: GNU License",
    ],
    install_requires=[
        "numpy","matplotlib"
    ],
    extras_require={
        "docs": ["sphinx","sphinx_rtd_theme","sphinxcontrib-bibtex"],
        "tests": ["pytest","py"]
    }
)

