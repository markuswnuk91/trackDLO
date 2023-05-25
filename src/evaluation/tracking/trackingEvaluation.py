import sys, os

try:
    sys.path.append(os.getcwd().replace("/src/evaluation/tracking", ""))
    from src.evaluation.evaluation import Evaluation

    # registration algorithms
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
except:
    print("Imports for class TrackingEvaluation failed.")
    raise


class TrackingEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {
            "spr": {
                "trackingErrors": [],
                "computationTimes": [],
                "geometricErrors": [],
                "T": [],
            }
        }

    def setupEvaluationCallback(
        self, classHandle, result, visualize=True, saveImages=True
    ):
        if isinstance(classHandle) == StructurePreservedRegistration:
            fig, ax = setupVisualization(classHandle.Y.shape[1])
            raise NotImplementedError
            return partial(
                visualizationCallbackTracking,
                fig,
                ax,
                classHandle,
                savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/bldoReconstruction/test/",
            )
        else:
            raise NotImplementedError

    def runSPR(
        self,
        PointClouds,
        XInit,
        iterationsUntilUpdate,
        sprParameters,
        vis=True,
        saveResults=True,
        saveImages=False,
    ):
        result = {}
        spr = StructurePreservedRegistration(
            **{
                "X": XInit,
                "Y": PointClouds[0],
            }
        )
        callback = self.setupEvaluationCallback(
            spr, result, visualize=True, saveImages=True
        )
        spr.register(callback)
        return result
