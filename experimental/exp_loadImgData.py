import os, sys

try:
    # image io imports
    import imageio.v3 as iio
except:
    print("\n---\nThis scipt requires imageio.v3!\n---\n")
# open cv imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
except:
    print("Imports for experimental script to load image data failed.")
    raise

path = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/image_disparity_1.png"

try:
    # imageio
    im = iio.imread(path)
    print(im.shape)
except:
    print("\n---\nThis scipt requires imageio.v3!\n---\n")

# opencv
imgGray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
imgRGB = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2RGB)
plt.imshow(imgGray, cmap="gray")
plt.pause(5)
