#!/bin/bash

# Check if correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <height> <sex>"
    exit 1
fi

# Assign arguments to variables
HEIGHT=$1
SEX=$2

# Navigate to Depth_Estimation folder and run the script
cd Depth_Estimation || { echo "Directory Depth_Estimation not found!"; exit 1; }
python run.py || { echo "Error running Depth_Estimation script!"; exit 1; }

# Return to the original folder
cd .. || { echo "Error returning to the original folder!"; exit 1; }

# Run the body fat estimation script
python estimate-body-fat.py --height "$HEIGHT" --sex "$SEX" || { echo "Error running body fat estimation script!"; exit 1; }

exit 0