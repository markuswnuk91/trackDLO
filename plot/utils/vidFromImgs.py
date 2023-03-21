import os
import moviepy.video.io.ImageSequenceClip


def extract_integer(filename):
    return int(filename.split(".")[0].split("_")[-1])


image_folder = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/bldoReconstruction/test/"
save_folder = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/vids/"
filename = "test.mp4"
fps = 10

image_files = [
    os.path.join(image_folder, img)
    for img in sorted(os.listdir(image_folder), key=extract_integer)[:400]
    if img.endswith(".png")
][::2]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(save_folder + filename)
