from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()

setup(
    name="torch_inspector",
    version="0.1.0",
    author="Luis Denninger",
    author_email="l_denninger@uni-bonn.de",
    description="""
        The 'torch_inspector' package provides InspectorGadgets, a simple module
        you can register your PyTorch models with and inspect intermediate shapes,
        per-layer memory usage for backward and forward pass and per-layer latency.
        Moreover, it implements NaN and Inf checks and counts your (learnable) parameters. 
        This can help you to catch bugs, assess efficiency, get a better insight or 
        get a rought estimate for required resources.
    """,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LDenninger/torch-inspector",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)