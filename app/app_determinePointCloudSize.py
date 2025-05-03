import sys
import os
import cv2

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.evaluation.evaluation import Evaluation
except:
    raise ImportError

pathToConfigFile = "eval/tracking/evalConfigs/evalConfig_20230524_161235_ManipulationSequences_mountedWireHarness_arena.json"

dataSetPaths = [
    "data/darus_data_download/data/20230524_171237_ManipulationSequences_mountedWireHarness_modelY/",
    "data/darus_data_download/data/20230524_161235_ManipulationSequences_mountedWireHarness_arena/",
    "data/darus_data_download/data/20230807_162939_ManipulationSequences_mountedWireHarness_partial/",
]

if __name__ == "__main__":
    evaluationHandler = Evaluation(configFilePath=pathToConfigFile)
    for dataSetPath in dataSetPaths:
        for i in range(0,100):
            # load data
            rgbImage, disparityMap = evaluationHandler.getDataSet(i, dataSetPath)
            points, colors = evaluationHandler.preProcessor.calculatePointCloudFiltered_2D_3D(
                    rgbImage, disparityMap
                )
            print("Dataset: {}, size: {}".format(dataSetPath,points.shape) )