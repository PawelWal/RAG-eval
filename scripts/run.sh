#!/bin/bash

# File paths
DATA_DIR="../data/rag"
OUTPUT_DIR="../data/rag_generation"
PYTHON_SCRIPT="generate.py"

# Check if the data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Catalogue $DATA_DIR does not exist."
  exit 1
fi

for filepath in "$DATA_DIR"/*.csv; do

    if [ -f "$filepath" ]; then

        # Extract the dataset name from the filename (remove the .csv extension)
        filename=$(basename "$filepath")
        ds_name="${filename%.csv}"

        # Define the output file path
        output_filepath="$OUTPUT_DIR/${ds_name}.csv"

        # Check if the output file already exists
        if [ -f "$output_filepath" ]; then
            echo "Skipping: $ds_name (File $output_filepath already exists)"
            continue # Skip to the next iteration of the loop
        fi

        echo "=================================================="
        echo "Uruchamianie przetwarzania dla: $ds_name"
        echo "=================================================="

        # Run the Python script with the dataset name as an argument
        python "$PYTHON_SCRIPT" --ds-name "$ds_name"

    fi
done

echo "All datasets have been processed."
