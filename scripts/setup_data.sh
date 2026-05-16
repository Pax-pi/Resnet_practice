#!/bin/bash
set -e

# step 1, check the environment variable
if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "Error: Kaggle API Token has not been set."
    exit 1
fi

# step 2, check  kaggle is installed
if command -v kaggle &> /dev/null; then
    :
else
    echo "Kaggle is not installed, installing Kaggle through pip."
    pip install kaggle
fi

# step 3, use kaggle to download the dataset
cd ..
mkdir -p data/rsna
cd data/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge

# step 4, unzip it

unzip -q rsna-pneumonia-detection-challenge.zip

# step 5, delete the zip file

rm rsna-pneumonia-detection-challenge.zip