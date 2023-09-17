import sys, os
import matplotlib
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
except:
    print("Imports for Colors File failed.")
    raise

colors = {
    "red": [1, 0, 0],
    "blue": [0, 0, 1],
    "uniSAnthrazit": [62 / 255, 68 / 255, 76 / 255],
    "uniSBlue": [0 / 255, 81 / 255, 158 / 255],
    "uniSLightBlue": [0 / 255, 190 / 255, 255 / 255],
    "uniSLightBlue80": [51 / 255, 103 / 255, 255 / 255],
    "uniSLightBlue60": [102 / 255, 216 / 255, 255 / 255],
    "uniSLightBlue40": [153 / 255, 229 / 255, 255 / 255],
    "uniSLightBlue20": [204 / 255, 242 / 255, 255 / 255],
}

colorPalettes = {
    "viridis": plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps["viridis"],
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
    )
}
