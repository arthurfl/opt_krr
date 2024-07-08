from setuptools import setup, find_packages

setup(
    name="opt_krr",
    version="0.1.0",
    description="Pytorch-implemented kernel ridge regression HP optimization",
    author="Arthur France-Lanord",
    author_email="arthur.france-lanord AT sorbonne-universite DOT fr",
    url="https://github.com/arthurfl/opt_krr",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

