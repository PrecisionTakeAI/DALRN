"""
DALRN Setup Configuration
"""
from setuptools import setup, find_packages

setup(
    name="dalrn",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.30.1",
        "pydantic>=2.7.4",
        "numpy>=1.26.4",
        "web3>=6.20.1",
        "tenseal==0.3.16",
        "torch>=2.0.0",
        "networkx>=3.0",
    ],
    author="DALRN Team",
    description="Distributed Adaptive Learning & Resolution Network",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
)