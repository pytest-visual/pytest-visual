from setuptools import setup, find_packages

setup(
    name='pytest-unitvis',
    version='0.1',
    packages=find_packages(),
    entry_points={'pytest11': ['unitvis = unitvis.plugin']},
)
