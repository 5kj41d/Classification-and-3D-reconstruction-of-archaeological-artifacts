#!/bin/bash

# Motivation: Run the training script. If the script is interupted by the server, then start it again. This continues until program exit status 0 is raised.
# Note: Should be used with checkpoints, else the script will start all over
while true; do

    nohup srun -p batch --cpus-per-gpu=16 --gres=gpu:3 --mem 256G singularity exec --nv pytorch_24.02-py3.sif python3 ./RandomSearchGAN.py > output.log 2>&1

    if [ $? -eq 0 ]; then # $? is the status of the most recent execution which is the script
        echo "Job completed successfully."
        break
    else
        echo "Job terminated prematurely. Resubmitting in 5 minutes..."
        sleep 5m
    fi
done

# Detatch from shell to be able to quit the shell without quitting the script execution
disown -h
