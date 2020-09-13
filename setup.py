from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name="python_tools",
    version=0.1,
    author="Mytlogos",
    author_email="mytlogos@hotmail.de",
    url="https://github.com/mytlogos/python_tools",
    description="Small Python Programs with Django Web Interface ",
    long_description=open("README.md").read(),
    # Package info
    packages=find_packages(exclude=("test",)),
    install_requires=requirements,
)
