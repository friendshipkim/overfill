from setuptools import setup, find_packages

setup(
    name="overfill",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Woojeong Kim",
    author_email="kwj962004@gmail.com",
    description="A package for two-stage decoding of language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/friendshipkim/overfill",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)