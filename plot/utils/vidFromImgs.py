import os
import moviepy.video.io.ImageSequenceClip


def extract_integer(filename):
    return int(filename.split(".")[0].split("_")[1])


image_folder = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail"
fps = 20

image_files = [
    os.path.join(image_folder, img)
    for img in sorted(os.listdir(image_folder), key=extract_integer)
    if img.endswith(".png")
]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile("my_video.mp4")
