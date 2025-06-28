# setup.py
from setuptools import setup

setup(
    name='baptc_predict_api',
    version='1.0',
    install_requires=open('requirements.txt').read().splitlines(),
)
