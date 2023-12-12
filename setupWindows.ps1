# Check if Chocolatey is installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}

# Install Python3 using Chocolatey
Write-Host "Installing Python3..."
choco install python

# Install pip
Write-Host "Installing pip..."
python -m ensurepip --default-pip

# Install Jupyter Lab
Write-Host "Installing Jupyter Lab..."
pip install jupyterlab

# Install virtualenv (venv) for Python
Write-Host "Installing virtualenv for Python..."
pip install virtualenv

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv windowsPythonVenv

# Activate the virtual environment
Write-Host "Activating virtual environment..."
.\windowsPythonVenv\Scripts\Activate

# Install Python libraries via requirements.txt
Write-Host "Installing Python libraries from requirements.txt..."
pip install -r requirements.txt


Write-Host "Setup complete. ;)"



