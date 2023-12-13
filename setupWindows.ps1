# Check policy to run the script
function TestExecutionPolicy {
    $currentPolicy = Get-ExecutionPolicy
    return ($currentPolicy -eq "RemoteSigned") 
}

# Function to prompt user with command to set execution policy
function Prompt-SetExecutionPolicyCommand {
    Write-Host "The execution policy is not set to 'RemoteSigned' on your system."
    Write-Host "To enable it, run the following command with administrative privileges:"
    Write-Host "set-executionpolicy remotesigned"
}

# Check if the execution policy is set to "RemoteSigned"
if (-not (Test-ExecutionPolicy)) {
    Prompt-SetExecutionPolicyCommand # Prompt the user
    return # Exit 
}

# If the correct policy is enabled, then start install Chocolatey:

# Check if Chocolatey is installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}

# Install Python3 using Chocolatey - 3.11.7 works with todays (12/2023) Pytorch
$pythonVersion = (python --version 2>&1)
if ($pythonVersion -notlike "Python 3.11*") {
    Write-Host "Python version not compatible. Installing Python3.11.7..."
    choco install python --version=3.11.7 -y

    Write-Host "Python 3.11.7 was installed. Need to reboot and run the script again :)"
    return # Exit
}

# If the correct Python version was installed, then install the rest below:

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



