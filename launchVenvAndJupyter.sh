#!/bin/bash

# Use the current path for the source command
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the current virtual environment if available
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "$CURRENT_DIR/myvenv/bin/activate" ]; then
        source "$CURRENT_DIR/myvenv/bin/activate"
        echo "Activated virtual environment: $CURRENT_DIR/myvenv"
	echo "Starting Jupyter-lab"
        jupyter-lab
    fi
fi










