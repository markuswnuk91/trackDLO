import sys
import os

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.evaluation.evaluation import Evaluation
except:
    print("Imports for show image application failed.")
    raise
image_file_path = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_image_rgb.png"

if __name__ == "__main__":

    eval = Evaluation()
    fileName = os.path.basename(image_file_path)
    datasetFolderPath = os.path.dirname(os.path.dirname(image_file_path)) + "/"
    eval.showImage(fileName, datasetFolderPath)
