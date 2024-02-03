#!/bin/bash

# Directory containing the images
IMAGE_DIR="/mnt/c/Users/gs111064/Documents/Dissertation/Software/trackdlo/imgs/pointCloudProcessing"

# Percentage of the height to be cropped from the top
PERCENT_TOP=10

# Percentage of the height to be cropped from the bottom
PERCENT_BOTTOM=20

# Percentage of the width to be cropped from the left
PERCENT_LEFT=10

# Percentage of the width to be cropped from the right
PERCENT_RIGHT=10

# Process each image
for img in "$IMAGE_DIR"/*.{jpg,png}; do
    # Get image dimensions
    dimensions=$(identify -format "%wx%h" "$img")
    width=$(echo $dimensions | cut -d'x' -f1)
    height=$(echo $dimensions | cut -d'x' -f2)

    # Calculate new dimensions and offsets
    crop_width=$((width - (width * PERCENT_LEFT / 100) - (width * PERCENT_RIGHT / 100)))
    crop_height=$((height - (height * PERCENT_TOP / 100) - (height * PERCENT_BOTTOM / 100)))
    offset_x=$((width * PERCENT_LEFT / 100))
    offset_y=$((height * PERCENT_TOP / 100))

    # Crop the image
    mogrify -crop ${crop_width}x${crop_height}+${offset_x}+${offset_y} "$img"
done
