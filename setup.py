from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="alps2qutipplus",
    version="1.0.0",
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
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=[
        "matplotlib",
        "qutip",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


