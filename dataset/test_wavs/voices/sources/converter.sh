#!/bin/bash

# Loop through all .m4a files in the current directory
for file in *.m4a; do
    # Get the base name without extension
    base_name="${file%.m4a}"
    
    # Convert to WAV using ffmpeg
    ffmpeg -i "$file" -acodec pcm_s16le -ar 44100 -ac 2 "${base_name}.wav"
    
    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "Converted: $file to ${base_name}.wav"
    else
        echo "Failed to convert: $file"
    fi
done

echo "Conversion complete."