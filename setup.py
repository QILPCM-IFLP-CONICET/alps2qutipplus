from distutils.core import setup
from setuptools import find_packages

setup(
    name="Alps2Qutip",
    version="1.0",
    # packages=["alpsqutip"],
    packages=find_packages(),
    package_data={
        "alpsqutip": [
            "lib/models.xml",
            "lib/lattices.xml",
        ],
    },
    url="http://www.fisica.unlp.edu.ar/Members/matera/english-version/mauricio-materas-personal-home-page",
    license="LICENSE.txt",
    author="Juan Mauricio Matera",
    author_email="matera@fisica.unlp.edu.ar",
    description="Your project description",
    long_description=open("README.rst").read(),
    install_requires=[
        "matplotlib",
        "qutip",
    ],
)
