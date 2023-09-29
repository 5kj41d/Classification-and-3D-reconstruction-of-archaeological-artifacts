#!/bin/bash

# Check if a virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is activated. Installing libraries in the current environment."
else
    echo "No virtual environment is activated. Installing libraries in the system-wide environment."
fi

# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install Python (if not already installed)
if ! type python &> /dev/null; then
    echo "Installing Python..."
    sudo pacman -S --noconfirm python
else
	echo "Python is already installed. Proceeding..."
fi

# Install pip (if not already installed)
if ! type pip &> /dev/null; then
    echo "Installing pip..."
    sudo pacman -S --noconfirm python-pip
else
	echo "pip is already installed. Proceeding..."
fi

# Install Jupyter Lab (if not already installed)
if ! type jupyter-lab &> /dev/null; then
    echo "Installing Jupyter Lab..."
    pip install jupyterlab
else
	echo "Jupyter Lab is already installed. Proceeding..."
fi

# Activate the current virtual environment if available
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "$CURRENT_DIR/venv/bin/activate" ]; then
        source "$CURRENT_DIR/venv/bin/activate"
        echo "Activated virtual environment: $CURRENT_DIR/venv"
    fi
fi

# Install NumPy
pip install numpy

# Install scikit-learn
pip install scikit-learn

# Install scikit-image (skimage)
pip install scikit-image

# Install TensorFlow (CPU version)
pip install tensorflow

# Install matplotlib
pip install matplotlib

# Install requests
pip install requests

# Install Pillow
pip install Pillow

echo "Libraries installed successfully."
