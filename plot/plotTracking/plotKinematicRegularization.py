import sys
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
except:
    print("Imports for plotting kinematic regulatization for tracking chapter failed.")
    raise
runOpt = {"save": True, "runInitialization": False}
saveOpt = {
    "initializationResultPath": "data/plots/kinematicRegularization",
    "saveFolderPath": "imgs/kinematicRegularization",
    "dpi": 300,
}

evalConfigPath = "plot/plotTracking/config.json"
filePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120351_790481_image_rgb.png"
if __name__ == "__main__":
    # load point set
    eval = TrackingEvaluation(evalConfigPath)
    fileName = os.path.basename(filePath)
    dataSetFolderPath = os.path.dirname(os.path.dirname(filePath)) + "/"
    (Y, _) = eval.getPointCloud(fileName, dataSetFolderPath)
    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=Y, color=[1, 0, 0], alpha=0.1, size=1)
    mask = np.ones(len(Y), dtype=bool)
    mask[300:1000] = False
    plotPointSet(ax=ax, X=Y[mask], color=[0, 0, 1], alpha=0.1, size=1)

    # get inital configuration
    if runOpt["runInitialization"]:
        initializationResult = eval.runInitialization(dataSetFolderPath, fileName)
        eval.saveWithPickle(
            data=initializationResult,
            filePath=os.path.join(
                saveOpt["initializationResultPath"],
                "initializationResult.pkl",
            ),
            recursionLimit=10000,
        )
    else:
        # save correspondance estimation results
        initializationResult = eval.loadResults(
            os.path.join(
                saveOpt["initializationResultPath"],
                "initializationResult.pkl",
            )
        )
    X_init = initializationResult["localization"]["X"]
    modelParameters = initializationResult["modelParameters"]
    model = eval.generateModel(modelParameters)
    q_init = initializationResult["localization"]["q"]
    model.setGeneralizedCoordinates(q_init)
    X_init = model.getCartesianBodyCenterPositions()
    plotPointSet(
        ax=ax, X=X_init, size=10, markerStyle="o", edgeColor=[0, 0, 1], color=[1, 1, 1]
    )

    # run tracking (GMM)
    # eval.config["cpdParameters"]["alpha"] = 0
    # cpd = CoherentPointDrift(
    #     **{"X": X_init, "Y": Y[mask]}, **eval.config["cpdParameters"]
    # )
    # cpd.registerCallback(eval.getVisualizationCallback(cpd))
    # cpd.register()

    # run tracking Regularized
    kinematicModel = KinematicsModelDart(model.skel.clone())
    kpr = KinematicsPreservingRegistration(
        Y=Y[mask], qInit=q_init, model=kinematicModel, **eval.config["kprParameters"]
    )
    kpr.registerCallback(eval.getVisualizationCallback(kpr))
    kpr.register()
    # visualizat point set
    plt.show(block=True)
    print("Done.")
