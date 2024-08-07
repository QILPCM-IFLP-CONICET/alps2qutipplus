from distutils.core import setup

setup(
    name="Alps2Qutip",
    version="1.0",
    packages=["alpsqutip"],
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
