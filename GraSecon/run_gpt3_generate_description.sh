#!/bin/bash

# Define the python script and log file paths
PYTHON_SCRIPT="gpt3_generate_description_imagenet1k.py"
LOG_FILE="gpt3_generate_description_imagenet1k.log"

# Run the python script in the background with unbuffered output
nohup python -u $PYTHON_SCRIPT | tee $LOG_FILE > /dev/null 2>&1 &
PID=$!

# Print the PID and log location for reference
echo "Script is running in the background with PID: $PID"
echo "Log output is being saved to: $LOG_FILE"
