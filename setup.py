#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

# Read the version from __init__.py
with open(os.path.join("src", "llm_fallbacks", "__init__.py"), "r") as f:
    version_match: re.Match[str] | None = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version: str = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

# Define requirements
requirements: list[str] = ["litellm>=1.0.0", "numpy>=1.20.0", "pandas>=1.3.0"]

setup(
    name="llm_fallbacks",
    version=version,
    description="A library for managing fallbacks for LLM API calls",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm_fallbacks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={"console_scripts": ["llm-fallbacks=llm_fallbacks.__main__:main"]},
    keywords="llm, ai, fallbacks, litellm",
)
