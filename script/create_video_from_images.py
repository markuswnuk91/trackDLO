import cv2
import os
import re
# Folder with images (e.g. img_0001.png, img_0002.png, ...)
image_folder = "data/eval/tracking/plots/trackingResults/20230524_161235_ManipulationSequences_mountedWireHarness_arena/kpr"

output_video = "data/vids/20230524_161235_ManipulationSequences_mountedWireHarness_arena_kpr_refined.mp4"
fps = 20

def frame_number(filename):
    match = re.search(r"frame_(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        return float("inf")  # put weird files at the end
    
# Get sorted list of images
images = sorted(
    [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
    key=frame_number
)

print("Sorted images:")
for img in images:
    print(img)

# Read first image to get size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add images to video
for img in images:
    frame = cv2.imread(os.path.join(image_folder, img))
    video.write(frame)

# video.release()
print("Video saved as", output_video)