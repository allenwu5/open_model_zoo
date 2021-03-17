"""
Based on https://packaging.python.org/tutorials/packaging-projects/
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open-model-zoo",
    version="0.0.1",
    author="",
    author_email="",
    description="OpenVINO Model ZOO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allenwu5/open_model_zoo",
    project_urls={
        "Bug Tracker": "https://github.com/allenwu5/open_model_zoo",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
)
