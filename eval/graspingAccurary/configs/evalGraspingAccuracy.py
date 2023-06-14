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

if __name__ == "__main__":
    # load model
    bdloModel = eval.generateModel(
        dataSetPath, eval.config["modelGeneration"]["numSegments"]
    )
    # determine initial configuration
    if runExperiment:
        if not loadInitialStateFromResult:
            # setup result file
            result = setupResultTemplate(dataSetPath)
            eval.results.append(result)
            runInitialization(dataSetPath, frame)
            runModelGeneration(dataSetPath)
            runInitialLocalization(dataSetPath, initialFrame)
        else:
            results = eval.loadResults(resultFilePath)
            eval.results = results
        
    # run tracking
    trackingResults = eval.(initalState)
    # evaluate results