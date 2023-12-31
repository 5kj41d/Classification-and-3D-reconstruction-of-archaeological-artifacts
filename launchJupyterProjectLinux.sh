#!/bin/bash

# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Current directory: $CURRENT_DIR"


# Activate the current virtual environment if available
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "$CURRENT_DIR/linuxPythonVenv/bin/activate" ]; then
        source "$CURRENT_DIR/linuxPythonVenv/bin/activate"
        echo "Activated virtual environment: $CURRENT_DIR/linuxPythonVenv"
	echo "Starting Jupyter-lab"
        nohup jupyter lab &
    fi
    else 
        nohup jupyter lab &
fi
else 
	echo "Could not launch... Check if the venv is intact"










