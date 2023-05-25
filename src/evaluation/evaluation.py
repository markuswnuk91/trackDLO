import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dartpy as dart
from scipy.spatial import distance_matrix
from functools import partial

try:
    sys.path.append(os.getcwd().replace("/src/evaluation", ""))
    # input data porcessing
    from src.sensing.preProcessing import PreProcessing
    from src.sensing.dataHandler import DataHandler

    # topology extraction
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.localization.downsampling.l1median.l1Median import L1Median

    # model generation
    from src.simulation.bdlo import BranchedDeformableLinearObject

    # initial localization
    from src.localization.bdloLocalization import (
        BDLOLocalization,
    )

    # tracking
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart

    # visualization
    from src.visualization.plot3D import *

except:
    print("Imports for evaluation class failed.")
    raise


class Evaluation(object):
    def __init__(self, pathToConfigFile, *args, **kwargs):
        self.evalConfigPath = pathToConfigFile
        self.dataHandler = DataHandler()
        self.evalConfig = self.dataHandler.loadFromJson(self.evalConfigPath)
