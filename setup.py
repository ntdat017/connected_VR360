# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='connectedvr360',
    platforms=['any'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version='1.0.0',
    summary='A Python package for fast and robust Connected VR360',
    description=long_description,
    description_content_type='text/markdown',
    author='Dat Nguyen Thanh',
    author_email='ntdat017@gmail.com',
    license="GPL",
    # url='https://github.com/',
    # download_url='https://github.com/',
    keywords=['panorama', '360', 'feature-matching'],
    classifiers=[
        ("License :: OSI Approved :: "
        "GNU General Public License v3 or later (GPLv3+)")
    ],
    # install_requires=['torch', 'py360convert', 'matplotlib'],
    entry_points={'console_scripts': ['connected_vr360=connectedvr360.main:main']}
)