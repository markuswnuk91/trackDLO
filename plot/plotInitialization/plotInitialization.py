import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.dartVisualizer import DartScene
    from src.visualization.plot3D import *
    from src.visualization.colors import *
    from src.visualization.plotUtils import scale_axes_to_fit
except:
    print("Imports for plotting tolology extraction failed.")
    raise
runOpt = {"save": True, "runTopologyExtraction": False, "blockAfterPlotting": True}
visOpt = {"visLocalizationIterations": True, "plotDartVis": False}
saveOpt = {
    "localizationResultSavePath": "data/plots/initialization",
    "imageSavePath": "imgs/initialization",
    "dpi": 300,
}
relFilePaths = [
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png",
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120112_746315_image_rgb.png",
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120436_605547_image_rgb.png",
]
relFilePath = relFilePaths[1]  # choose the configuration
configPath = "plot/plotInitialization/evalConfig.json"
styleOpt = {
    "iterationsToPlot": [90],
    # "iterationsToPlot": [0, 1, 5, 29],
    "templatePointColor": [0, 0, 1],
    "targetPointColor": [1, 0, 0],
    "elevation": 25,
    "azimuth": 125,
    "correspondanceColor": [0.7, 0.7, 0.7],
    "correspondacenAlpha": 0.8,
    "targetPointColor": [1, 0, 0],
    "targetLineColor": [1, 0, 0],
    "targetPointSize": 0.3,
    "targetLineWidth": 0.1,
    "targetPointAlpha": 0.3,
    "targetLineAlpha": 0.3,
    "templatePointColor": [0, 0, 1],
    "templateLineColor": [0, 0, 1],
    "templatePointSize": 0.01,
    "templateLineWidth": 1.5,
    "templatePointAlpha": 0.9,
    "templateLineAlpha": 0.9,
    "correspondenceTemplatePointSize": 20,
    "correspondenceTargetPointSize": 0.1,
    "eyePosition_DART": [2.5, 0, 1.5],
    "colorPalette": thesisColorPalettes["viridis"],
}

if __name__ == "__main__":
    eval = InitialLocalizationEvaluation(configFilePath=configPath)
    # set data set paths
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetPath = os.path.dirname(dataFolderPath) + "/"
    frame = eval.getFrameFromFileName(dataSetPath, fileName)
    save_name = fileName.split(".")[0]
    # configure initial configuration
    model, modelParameters = eval.getModel(dataSetPath)

    q_root_init = modelParameters["branchSpecs"][0]["rootJointRestPositions"]
    xyzOffset = np.array(([0.4, 0.4, 0.5]))
    rotOffset = [np.pi / 2, -np.pi / 2, 0]
    q_root_init[3:6] = xyzOffset
    q_root_init[:3] = model.convertExtrinsicEulerAnglesToBallJointPositions(
        rotOffset[0], rotOffset[1], rotOffset[2]
    )
    modelParameters["branchSpecs"][0]["rootJointRestPositions"] = q_root_init
    model = eval.generateModel(modelParameters)

    if runOpt["runTopologyExtraction"]:
        # perfrom topology extraction & correspondance estimation for input data set
        pointCloud = eval.getPointCloud(
            frame, dataSetPath, segmentationMethod="standard"
        )
        Y = pointCloud[0]
        extractedTopology = eval.extractMinimumSpanningTreeTopology(Y, model)[
            "extractedTopology"
        ]
        localizationResult, _ = eval.runInitialLocalization(
            pointSet=Y,
            extractedTopology=extractedTopology,
            bdloModelParameters=modelParameters,
            visualizeCorresponanceEstimation=True,
            visualizeIterations=visOpt["visLocalizationIterations"],
            visualizeResult=False,
            block=False,
            closeAfterRunning=True,
            logResults=True,
        )
        eval.saveWithPickle(
            data=localizationResult,
            filePath=os.path.join(
                saveOpt["localizationResultSavePath"],
                "localizationResult" + "_" + save_name + ".pkl",
            ),
            recursionLimit=10000,
        )
    else:
        # save correspondance estimation results
        localizationResult = eval.loadResults(
            os.path.join(
                saveOpt["localizationResultSavePath"],
                "localizationResult" + "_" + save_name + ".pkl",
            )
        )

    # plot iterations
    Y_adjacencyMatrix = localizationResult["extractedTopology"].adjacencyMatrix
    Y = localizationResult["extractedTopology"].X
    qInterations = []
    qInterations.append(model.getGeneralizedCoordinates())
    for i in styleOpt["iterationsToPlot"]:
        qInterations.append(localizationResult["qLog"][i])
    iterations = np.array(styleOpt["iterationsToPlot"]) + 1
    np.insert(iterations, 0, 0)
    set_text_to_latex_font(scale_axes_labelsize=2)
    for i, q in enumerate(qInterations):
        fig, ax = setupLatexPlot3D()
        plotGraph3D(
            ax=ax,
            X=Y,
            adjacencyMatrix=Y_adjacencyMatrix,
            pointColor=styleOpt["targetPointColor"],
            lineColor=styleOpt["targetLineColor"],
            pointSize=styleOpt["targetPointSize"],
            lineWidth=styleOpt["targetLineWidth"],
            pointAlpha=styleOpt["targetPointAlpha"],
            lineAlpha=styleOpt["targetLineAlpha"],
        )
        X, X_adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix(q)

        # plotGraph3D(
        #     ax=ax,
        #     X=X,
        #     adjacencyMatrix=X_adjacencyMatrix,
        #     pointColor=styleOpt["templatePointColor"],
        #     lineColor=styleOpt["templateLineColor"],
        #     pointSize=styleOpt["templatePointSize"],
        #     lineWidth=styleOpt["templateLineWidth"],
        #     pointAlpha=styleOpt["templatePointAlpha"],
        #     lineAlpha=styleOpt["templateLineAlpha"],
        # )
        (_, _, branchCorrespondanceMatrix) = model.getBranchCorrespondancesForJoints()
        # make adjacency matrix symmetric
        X_adj_sym = ((X_adjacencyMatrix + X_adjacencyMatrix.T) > 0).astype(int)
        plotBranchWiseColoredGraph3D(
            ax=ax,
            positions3D=X,
            adjacencyMatrix=X_adj_sym,
            branchCorrespondanceMatrix=branchCorrespondanceMatrix,
            colorPalette=None,
            lineWidth=styleOpt["templateLineWidth"],
            pointSize=styleOpt["correspondenceTemplatePointSize"],
        )

        # correspondanes
        Y_target = localizationResult["YTarget"]
        X_sample = model.getSamplePositionsFromLocalCoordinates(
            S=localizationResult["S"]
        )
        C = localizationResult["C"]
        plotCorrespondances3D(
            ax=ax,
            X=X_sample,
            Y=Y_target,
            C=C,
            xSize=0,
            ySize=styleOpt["correspondenceTargetPointSize"],
            correspondanceColor=styleOpt["correspondanceColor"],
            lineAlpha=styleOpt["correspondacenAlpha"],
        )

        scale_axes_to_fit(ax=ax, points=Y_target)
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(-0.3, 0.4)
        ax.set_zlim(-0.05, 0.65)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks([0.2, 0.4, 0.6, 0.8])
        ax.yaxis.set_ticks([-0.2, 0, 0.2, 0.4])
        ax.zaxis.set_ticks([0, 0.2, 0.4, 0.6])
        ax.view_init(azim=styleOpt["azimuth"], elev=styleOpt["elevation"])
        if runOpt["save"]:
            # check if directory exists
            path = os.path.join(saveOpt["imageSavePath"], save_name)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(
                os.path.join(
                    path,
                    "initialLocalization_" + str(iterations[i-1]) + ".pdf",
                ),
                bbox_inches="tight",
                pad_inches=0.5,
                # dpi=saveOpt["dpi"],
            )
        if visOpt["plotDartVis"]:
            model.setBranchColorsFromColorPalette(styleOpt["colorPalette"])
            dartScene = DartScene(model.skel, q, loadRobot=True, loadCell=True)
            dartScene.addPointCloud(points=Y, colors=[1, 0, 0])
            # dartScene.setModelColor([0, 0, 1])
            dartScene.saveFrame(
                savePath=os.path.join(
                    saveOpt["imageSavePath"],
                    save_name,
                    "dart_initialLocalization_" + str(iterations[i]),
                ),
                eye=styleOpt["eyePosition_DART"],
                center=[0, 0, 0],
                up=[0, 0, 1],
            )
        plt.show(block=runOpt["blockAfterPlotting"])
        plt.pause(0.5)
        plt.close("all")
    print("Done.")
