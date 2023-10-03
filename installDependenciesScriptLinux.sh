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

# Function to install Python venv on Manjaro
install_manjaro_python_venv() {
    echo "Installing Python venv on Manjaro..."
    sudo pacman -S --noconfirm python-virtualenv  # Install Python venv package for Manjaro
}

# Function to create and activate a virtual environment
create_and_activate_venv() {
    if [ ! -d "$CURRENT_DIR/linuxPythonVenv" ]; then
        echo "Creating a virtual environment..."
        python -m venv "$CURRENT_DIR/linuxPythonVenv"
        else 
            echo "Removing exising venv and creating a new..."
            rm -rf "$CURRENT_DIR/linuxPythonVenv"
            python -m venv "$CURRENT_DIR/linuxPythonVenv"
    fi

    echo "Activating virtual environment..."
    source "$CURRENT_DIR/linuxPythonVenv/bin/activate"
}

# Determine the Linux distribution. The -f is testing for existing file. 
if [ -f /etc/manjaro-release ]; then
    echo "Detected Manjaro Linux"
    install_manjaro
    install_manjaro_python_venv
elif [ -f /etc/lsb-release ]; then
    DISTRO=$(lsb_release -si)
    if [ "$DISTRO" == "Ubuntu" ]; then
        echo "Detected Ubuntu Linux"
        install_ubuntu
        install_ubuntu_python_venv
    else
        echo "Unsupported Linux distribution: $DISTRO"
        exit 1
    fi
else
    echo "Unsupported Linux distribution"
    exit 1
fi

# Create and activate a virtual environment
create_and_activate_venv

# Install project dependencies from requirements.txt
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install -r "$CURRENT_DIR/requirements.txt"
fi

echo "Libraries installed successfully."
