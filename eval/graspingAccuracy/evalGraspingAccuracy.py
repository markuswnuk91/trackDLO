import sys
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.evaluation import Evaluation

    # tracking algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.tracking.krspr.krspr import (
        KinematicRegularizedStructurePreservedRegistration,
    )
    from src.tracking.krcpd.krcpd import (
        KinematicRegularizedCoherentPointDrift,
    )

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

global vis
global result
vis = True
save = True
runExperiment = True

# setup evalulation class
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__)) + "/evalConfigs/evalConfig.json"
)
eval = Evaluation(configFilePath=pathToConfigFile)
# set file paths
dataSetPath = eval.config["dataSetPaths"][eval.config["dataSetToLoad"]]
dataSetName = eval.config["dataSetPaths"][0].split("/")[-2]
resultFolderPath = "data/eval/tracking/" + dataSetName + "/"
resultFileName = "result"
resultFilePath = resultFolderPath + resultFileName + ".pkl"


def evaluateGraspingAccuracy(dataSetPath, frame):
    graspingAccuracyResult = {}

    # run different tracking algorithms
    trackingResult = eval.runTracking(
        dataSetPath=dataSetPath,
        method="cpd",
        startFrame=frame,
        endFrame=frame,
        checkConvergence=True,
    )
    print(trackingResult)
    return graspingAccuracyResult


if __name__ == "__main__":
    initializationFrame = eval.config["initialFrame"]
    evaluationFrame = initializationFrame + 5
    # initialize on the first frame of the data set
    if runExperiment:
        initializationResult = eval.runInitialization(
            dataSetPath, initializationFrame, visualize=False
        )

    # evaluate grasping accuracy for different algorithms
    graspingAccuracyResult = evaluateGraspingAccuracy(dataSetPath, evaluationFrame)

    # else:
    #     results = eval.loadResults(resultFilePath)
    #     eval.results = results

    # evaluate results
