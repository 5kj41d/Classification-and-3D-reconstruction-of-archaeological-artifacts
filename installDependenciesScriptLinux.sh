# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $CURRENT_DIR

: << COMMENT
# Activate the current virtual environment if available
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
COMMENT

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

# Determine the Linux distribution. The -f is testing for existing file. 
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

# Install project dependencies from requirements.txt
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt"
fi

echo "Libraries installed successfully."
