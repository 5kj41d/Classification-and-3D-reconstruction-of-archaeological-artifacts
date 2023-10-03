# Classification-and-3D-reconstruction-of-archaeological-artifacts
To be written...


# Installation and Launch 
This script automates the installation of dependencies and the launches the project in Jupyter lab.

## Table of Contents 
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage) 
- [Troubleshooting](#troubleshooting)

## Introduction

This installation process works for:
	Linux Manjaro and Ubuntu.

## Installation

To install and setup the project run the following commands:

First clone the project from Github:
```bash 
	git clone git@github.com:5kj41d/Classification-and-3D-reconstruction-of-archaeological-artifacts.git
```
Locate to the project folder:
```bash 
	cd Classification-and-3D-reconstruction-of-archaeological-artifacts
```
To install the dependencies for the Jupyter project run:
```bash 
	./installDependenciesScriptLinux.sh
```

## Usage 

To run the project use the following command in the root project folder:
```bash 
	./launchVenvAndJupyterLinux.sh
```

## Troubleshooting

If you encounter any issues during installation or launch, consider the following:

Make sure you are located in the root project folder Classification-and-3D-reconstruction-of-archaeological-artifacts
- **Issue**: If you you cannot enter the project folder with provided cd command:
  - **Solution**: Locate the project folder where you installed it and open a terminal inside. 
```bash 
	sudo find / -type d -name Classification-and-3D-reconstruction-of-archaeological-artifacts
```

- **Issue**: If the virtual enviroment wont activate:
  - **Solution**: Enter the linuxPythonVenv/bin folder and run 
```bash 
	source activate
```
Then try to install again.

- **Issue**: If the installation script is finding a virtual enviroment but installing on the system path can be handled bu removing the 
linuxPythonVenv folder and create it again:
  - **Solution**: Try following commands:
Remove:
```bash 
	rm -rf linuxPythonVenv 
```
Create: 
```bash 
	python -m venv linuxPythonVenv 
```
Activate: 
```bash 
	source linuxPythonVenv/bin/activate
```

