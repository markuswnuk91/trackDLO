#!/bin/bash

# Directory containing the images
IMAGE_DIR="/mnt/c/Users/gs111064/Documents/Dissertation/Software/trackdlo/imgs/pointCloudProcessing"

# Percentage of the height to be cropped from the bottom
PERCENT_TO_CROP=10

# Process each image
for img in "$IMAGE_DIR"/*.{jpg,png}; do
    # Get image height
    height=$(identify -format "%h" "$img")

    # Calculate new height
    new_height=$((height - (height * PERCENT_TO_CROP / 100)))

    # Crop the image
    mogrify -crop x$new_height+0+0 "$img"
done
