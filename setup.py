from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
name="aura-dep-agent",
version="0.1.9",
description="AURA: an automated dependency update agent that searches for safe modernized requirements using your existing validation.",
long_description=readme,
long_description_content_type="text/markdown",
author="AURA_AUTHORS",
author_email="",
url="https://github.com/Bhargavvvv2912/AURA",
project_urls={
"Source": "https://github.com/Bhargavvvv2912/AURA",
"Tracker": "https://github.com/Bhargavvvv2912/AURA/issues",
},
license="MIT",
packages=find_packages(),
python_requires=">=3.10",
install_requires=[
"google-genai",
"google-api-core>=2.0.0",
"pypi-simple",
"tenacity",
"packaging",
],
entry_points={
"console_scripts": [
"aura=aura.cli:main",
],
},
classifiers=[
"Programming Language :: Python",
"Programming Language :: Python :: 3",
"Programming Language :: Python :: 3.10",
"Programming Language :: Python :: 3.11",
"License :: OSI Approved :: MIT License",
"Operating System :: OS Independent",
"Intended Audience :: Developers",
"Topic :: Software Development",
"Topic :: Software Development :: Libraries",
"Topic :: Software Development :: Testing",
],
)