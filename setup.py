import os
from typing import List

from setuptools import find_packages, setup


def get_requirements() -> List[str]:
    return [
        "pytest>=7.0.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "numpy>=1.17.0",
        "pandas>=2.0.0",
        "graphviz>=0.17",
        "Pillow>=8.0.0",
        "pydantic>=2.0.0",
    ]


version = os.getenv("RELEASE_VERSION", "v0.0.1")
if version.startswith("v"):
    version = version[1:]

print(f"Building pytest-visual {version}")

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
    url="https://github.com/pytest-visual/pytest-visual",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
