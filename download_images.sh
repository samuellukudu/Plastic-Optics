#!/bin/bash

# Array of file indices
files=(0 1 2 3 4 5 6)

# Loop through the file indices and download each file
for i in "${files[@]}"; do
    wget "https://zenodo.org/records/7991872/files/images$i.zip" -O "images$i.zip"
done

