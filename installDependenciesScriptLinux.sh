# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $CURRENT_DIR

# Function to install Python, pip, and Jupyter Lab on Manjaro
install_manjaro() {
    echo "Installing Python..."
    sudo pacman -S --noconfirm python || { echo "Failed to install Python"; exit 1; }

    echo "Installing pip..."
    sudo pacman -S --noconfirm python-pip
}

# Function to install Python, pip, and Jupyter Lab on Ubuntu
install_ubuntu() {
    echo "Installing Python..."
    sudo apt-get update
    sudo apt-get install -y python3 || { echo "Failed to install Python"; exit 1; }

    echo "Installing pip..."
    sudo apt-get install -y python3-pip
}

# Function to install Python venv
install_ubuntu_python_venv() {
    echo "Installing Python venv..."
    sudo apt-get install -y python3-venv  # Install Python venv package
}

# Function to install Python venv
install_python_venv_manjaro() {
    echo "Installing Python venv..."
        sudo pacman -S --noconfirm python-virtualenv  # Install Python venv package for Manjaro
}

# Function to install Python venv on Manjaro
install_manjaro_python_venv() {
    echo "Installing Python venv on Manjaro..."
    sudo pacman -S --noconfirm python-virtualenv  # Install Python venv package for Manjaro
}


# Activate the current virtual environment if available
activate_virtual_python_enviroment() {
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "$CURRENT_DIR/linuxPythonVenv/bin/activate" ]; then
        source "$CURRENT_DIR/linuxPythonVenv/bin/activate"
        echo "Activated virtual environment: $CURRENT_DIR/linuxPythonVenv"
        echo "Virtual environment is activated. Installing libraries in the current environment."
    else
        echo "Could not activate the virtual environment..."
        exit 1
    fi
fi
}

# Determine the Linux distribution. The -f is testing for existing file. 
if [ -f /etc/manjaro-release ]; then
    echo "Detected Manjaro Linux"
    install_manjaro
    install_manjaro_python_venv
    activate_virtual_python_enviroment
elif [ -f /etc/lsb-release ]; then
    DISTRO=$(lsb_release -si)
    if [ "$DISTRO" == "Ubuntu" ]; then
        echo "Detected Ubuntu Linux"
        install_ubuntu
        install_ubuntu_python_venv
        activate_virtual_python_enviroment
    else
        echo "Unsupported Linux distribution: $DISTRO"
        exit 1
    fi
else
    echo "Unsupported Linux distribution"
    exit 1
fi

# Install project dependencies from requirements.txt
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt"
fi

echo "Libraries installed successfully."
