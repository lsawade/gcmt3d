"""Package __init__.py docstring
Let's see whether Sphinx will print this Package Docstring"""

# Declare namespace
__import__('pkg_resources').declare_namespace(__name__)

# Import packages:
from .source import CMTSource



# The function below I found on stackoverflow. It is used to import  all
# subpackages and modules, when gcmt3d is imported without specification of the
# subpackages

import importlib
import pkgutil
import sys

def import_submodules(package_name):
    """ Import all submodules of a module, recursively

    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]\


    From Stackoverflow: User:   kolypto
                        Posted: 01.08.2014
                        Found:

    Added by Lucas Sawade 27.02.2019


    """
    package = sys.modules[package_name]
    return {
        name: importlib.import_module(package_name + '.' + name)
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__)
    }

import_submodules(__name__)