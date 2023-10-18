from typing import List
from setuptools import setup, find_packages

def get_requirements() -> List[str]:
    return [
        'pytest>=7.0.0',
    ]

setup(
    name='pytest-unitvis',
    version='0.0.2',
    packages=find_packages(),
    entry_points={'pytest11': ['unitvis = unitvis.plugin']},

    python_requires='>=3.8',
    install_requires=get_requirements(),

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    license='MIT',
    license_files=['LICENSE'],
    author='Kristjan Kongas',

    url='https://github.com/kongaskristjan/pytest-unitvis',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
