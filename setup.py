# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='neural_style',
    version='0.1',
    namespace_packages=['neural_style'],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'neural_style = neural_style.app:main',
        ],
    }
)
