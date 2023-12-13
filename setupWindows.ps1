# Check if Chocolatey is installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}

# Install Python3 using Chocolatey - 3.11.7 works with todays (12/2023) Pytorch
Write-Host "Installing Python3.11.7..."
choco install python --version=3.11.7 -y

Write-Host "Python 3.11.7 was installed. Need to reboot and run the script again :)"

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



