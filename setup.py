import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fishchips",
    version="0.0.1",
    author="Zack Li",
    author_email="zq@princeton.edu",
    description="Easy, extensible fisher forecasts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xzackli/fishchips-public",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
