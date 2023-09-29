#!/bin/bash

# Check if a virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is activated. Installing libraries in the current environment."
else
    echo "No virtual environment is activated. Installing libraries in the system-wide environment."
fi

# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to install Python, pip, and Jupyter Lab on Manjaro
install_manjaro() {
    echo "Installing Python..."
    sudo pacman -S --noconfirm python

    echo "Installing pip..."
    sudo pacman -S --noconfirm python-pip

    echo "Installing Jupyter Lab..."
    pip install jupyterlab
}

# Function to install Python, pip, and Jupyter Lab on Ubuntu
install_ubuntu() {
    echo "Installing Python..."
    sudo apt-get update
    sudo apt-get install -y python3

    echo "Installing pip..."
    sudo apt-get install -y python3-pip

    echo "Installing Jupyter Lab..."
    pip install jupyterlab
}

# Determine the Linux distribution
if [ -f /etc/manjaro-release ]; then
    echo "Detected Manjaro Linux"
    install_manjaro
elif [ -f /etc/lsb-release ]; then
    DISTRO=$(lsb_release -si)
    if [ "$DISTRO" == "Ubuntu" ]; then
        echo "Detected Ubuntu Linux"
        install_ubuntu
    else
        echo "Unsupported Linux distribution: $DISTRO"
        exit 1
    fi
else
    echo "Unsupported Linux distribution"
    exit 1
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
