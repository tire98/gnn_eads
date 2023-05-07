
from setuptools import setup, find_packages
setup(
    name='gnn_eads',
    version='1.0.0',
    description='Module for the developement of graph neural networks for the prediction of adsorption energies.',
    author='Tim Renningholtz',
    author_email='trenningholtz@iciq.es',
    packages=find_packages(),  # includ:e all packages under the project root directory 
    install_requires=[],  # list of dependencies 
    extras_require={},  # optional dependencies 
    entry_points={},  # entry points for command line scripts 
)