# Classification-and-3D-reconstruction-of-archaeological-artifacts

## Archaeological Artefact Classification using Convolutional Neural Network (CNN) and Vision Transformer
### Overview
This project aims to classify archaeological artefacts using two state-of-the-art models: Convolutional Neural Network (CNN) and Vision Transformer (ViT).
The primary goal is to explore strengths and weaknesses of each approach in the context of archaeological artefact classification. 
Additionally, in the future we want to focus on object regeneration i.e. the possibilities of reconstructing damaged artefacts. 

#### Convolutional Neural Network (CNN)

**Implementation details**
- Input layer: RGB images of archaeological artefacts.
- Convolutional Layers: Capture local features.
- Pooling layers: Reduce spatial dimensions.
- Fully Connected Layers: Make classification decisions.

#### Vision Transformer (Vit)

**Implementation details**
- Input Embedding: Splits the image into patches.
- Positional Encoding: Adds spatial information to the patches.
- Add learnable class/token parameter to the input sequence.
- Transformer Body: Apply self-attention mechanism (Multi-head attention) for feature connections.
- Classification Head: Makes final predictions.

### Dataset
The dataset is a diverse collection of archaeological artefacts recieved from the DIME database maintained and provided by Moesgaard Museum Aarhus, Denmark. 
Annotation for image classes is provided.


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
	./launchJupyterProjectLinux.sh
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

- **Issue**: If the error of exernal management is occuring may be due to missing Python Venv package. 
  - **Solution**: If the script do not install this, this can be done manually. Then try install process again. 

