#!/bin/bash

# Check if the source directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <source_directory>"
  exit 1
fi

# Define the source and target directories
SOURCE_DIR="$1"
TARGET_DIR="wavs-16k"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Loop through all WAV files in the source directory
for file in "$SOURCE_DIR"/*_48k_*.wav; do
  if [ -e "$file" ]; then
    # Extract the filename without the path
    filename=$(basename "$file")
    
    # Replace "48k" with "16k" in the filename
    new_filename=$(echo "$filename" | sed 's/48k/16k/')
    
    # Define the full path for the new file
    new_filepath="$TARGET_DIR/$new_filename"
    
    # Convert the sample rate to 16000 Hz using SoX with very high quality
    sox "$file" "$new_filepath" rate -v 16000
    
    echo "Converted $file to $new_filepath"
  fi
done

echo "Conversion completed."
