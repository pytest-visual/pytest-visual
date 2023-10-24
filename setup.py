import os
from typing import List

from setuptools import find_packages, setup


def get_requirements() -> List[str]:
    return [
        "pytest>=7.0.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "pandas>=2.0.0",
    ]


version = os.getenv("RELEASE_VERSION", "0.0.1")
assert version is not None, "RELEASE_VERSION environment variable must be set, this is typically done by the release pipeline."

setup(
    name="pytest-visual",
    version=version,
    packages=find_packages(),
    entry_points={"pytest11": ["visual = visual.interface"]},
    python_requires=">=3.8",
    install_requires=get_requirements(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    author="Kristjan Kongas",
    url="https://github.com/kongaskristjan/pytest-visual",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
