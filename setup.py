#!/usr/bin/env python
# Copyright (c) Institute of Artificial Intelligence (TeleAI), China Telecom, 2025. All Rights Reserved.

import re
import setuptools


def get_package_dir():
    pkg_dir = {
        "ruyi.tools": "tools"
    }
    return pkg_dir

def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs

def get_ruyi_version():
    with open("ruyi/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setuptools.setup(
    name="ruyi",
    version=get_ruyi_version(),
    author="Institute of Artificial Intelligence (TeleAI), China Telecom.",
    url="",
    package_dir=get_package_dir(),
    packages=setuptools.find_packages(exclude=("tools")) + list(get_package_dir().keys()),
    python_requires=">=3.8",
    install_requires=get_install_requirements(),
    setup_requires=["wheel"],  # avoid building error when pip is not updated
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,  # include files in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    project_urls={
        "Documentation": "",
        "Source": "",
        "Tracker": "",
    },
    include_dirs=[],
    ext_modules=[],
    cmdclass={}
)
