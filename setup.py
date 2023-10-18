from typing import List
from setuptools import setup, find_packages

def get_requirements() -> List[str]:
    return [
        'pytest>=7.0.0',
    ]

setup(
    name='pytest-unitvis',
    version='0.0.1',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    entry_points={'pytest11': ['unitvis = unitvis.plugin']},
    install_requires=get_requirements(),
)
