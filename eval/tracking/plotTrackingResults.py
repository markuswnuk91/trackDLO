import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = TrackingEvaluation()

# script contol options
controlOpt = {
    "resultsToLoad": [0],  # 0: modelY, 1: partial, 2: arena
    "methods": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
    "frames": list(range(290, 650, 5)),
    # [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650], # for modelY
    # [5, 50, 100, 150, 200, 250, 300], or list(range(0, 300, 5)) # for partial
    # [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 495], # for arena
    # [TBD], # for arena
    # --------------------------------------------
    # save opts
    "saveInputs": False,
    "save": True,
    "saveFolder": "data/eval/tracking/plots/trackingResults",
    "saveName": "trackingResult",
    "verbose": True,
    # --------------------------------------------
    # 2D plots
    "generateResults_2D": False,
    "showResult_2D": False,
    "blockResultPlot_2D": True,
    # --------------------------------------------
    # 3D plots
    "generateResults_3D": True,
    "showResult_3D": False,
    "blockResultPlot_3D": True,
    "saveFormat_3D": ".pdf",
    "dpi": 300,
    # --------------------------------------------
    # Simulation
    "generateResults_Sim": True,
    "showResult_Sim": False,
    # --------------------------------------------
}

# style options
scriptColorPalette = thesisColorPalettes["viridis"]
styleOpt_2D = {
    "colorPalette": scriptColorPalette,
    "lineThickness": 5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
}
styleOpt_3D = {
    "colorPalette": scriptColorPalette,
    "modelPointSize": 90,
    "modelLineWidth": 3,
    "pointCloudSize": 30,
    "pointCloudColor": [1, 0, 0],
    "pointCloudAlpha": 0.2,
    "pointCloudDownsampleFactor": 2,
    "targetAlpha": 1,
    "elevation": 30,  # 30 for isometric view
    "azimuth": 45,  # 45 for isometric view
    "axLimX": [0.3, 0.7],
    "axLimY": [0.1, 0.6],
    "axLimZ": [0.35, 0.75],
    "elev": 83,
    "azim": 80,
    "zoom": 1.5,
}

styleOpt_Sim = {
    "colorPalette": scriptColorPalette,
    "plotPointCloud": False,
    "pointCloudColor": [1, 0, 0],
    "pointCloudPointSize": 1,
    "targetPointSize": 10,
    "pointCloudAlpha": 0.1,
    "boardColor": [1, 1, 1],
    "camEye": [0.4, 1.3, 1.3],
    "camCenter": [0.4, 0, 0],
    "camUp": [0, 0, 1],
}

resultFileName = "result.pkl"

resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]


def loadResult(filePath):
    _, file_extension = os.path.splitext(filePath)
    if file_extension == ".pkl":
        with open(filePath, "rb") as f:
            result = pickle.load(f)
    return result


def saveInputImage(dataSetResult, frame):
    dataSetPath = dataSetResult["dataSetPath"]
    rgbImg = eval.getDataSet(frame, dataSetPath)[0]
    # save Visualization
    filename = "input" + "_frame_" + str(frame)
    dataSetPath = dataSetResult["dataSetPath"]
    dataSetName = dataSetPath.split("/")[-2]
    folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, "inputs")
    if not os.path.exists(folderPath):
        os.makedirs(folderPath, exist_ok=True)
    savePath = os.path.join(folderPath, filename)
    eval.saveImage(rgbImg, savePath)
    if controlOpt["verbose"]:
        print(
            "Saved input frame {} for data set {} as {}".format(
                frame, dataSetName, savePath
            )
        )


def createResultPlot_2D(dataSetResult, frame, method):
    resultImg = eval.plotBranchWiseColoredTrackingResult2D(
        result=dataSetResult,
        frame=frame,
        method=method,
        colorPalette=styleOpt_2D["colorPalette"],
        lineThickness=styleOpt_2D["lineThickness"],
        circleRadius=styleOpt_2D["circleRadius"],
    )
    fig = eval.plotImageWithMatplotlib(resultImg)
    if controlOpt["showResult_2D"]:
        plt.show(block=controlOpt["blockResultPlot_2D"])
    else:
        plt.close(fig)
    # save Visualization
    if controlOpt["save"]:
        filename = (
            controlOpt["saveName"] + "_" + method + "_" + "_frame_" + str(frame) + "_2D"
        )
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, method)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        savePath = os.path.join(folderPath, filename)
        eval.saveImage(resultImg, savePath)
        if controlOpt["verbose"]:
            print(
                "Saved 2D result for frame {} for method {} as {}".format(
                    frame, method, savePath
                )
            )
    return


def createResultPlot_3D(dataSetResult, frame, method):
    fig, ax = setupLatexPlot3D()

    registrationResult = eval.findCorrespondingEntryFromKeyValuePair(
        dataSetResult["trackingResults"][method]["registrations"], "frame", frame
    )
    pointCloud = registrationResult["Y"]
    targets = registrationResult["T"]
    adjacencyMatrix = dataSetResult["trackingResults"][method]["adjacencyMatrix"]
    dataSetPath = dataSetResult["dataSetPath"]
    modelParameters = dataSetResult["trackingResults"][method]["modelParameters"]
    bdloModel = eval.generateModel(modelParameters)
    ax = eval.plotBranchWiseColoredTrackingResult3D(
        ax=ax,
        X=targets,
        bdloModel=bdloModel,
        colorPalette=styleOpt_3D["colorPalette"],
        lineWidth=styleOpt_3D["modelLineWidth"],
        pointSize=styleOpt_3D["modelPointSize"],
    )
    # downsample point cloud
    # Create a boolean mask that is True for every nth point
    mask = np.ones(len(pointCloud), dtype=bool)
    mask[:: styleOpt_3D["pointCloudDownsampleFactor"]] = (
        False  # Mark every nth point as False
    )

    # Apply the mask to X
    downsampledCloud = pointCloud[mask]
    plotPointSet(
        ax=ax,
        X=downsampledCloud,
        color=styleOpt_3D["pointCloudColor"],
        size=styleOpt_3D["pointCloudSize"],
        alpha=styleOpt_3D["pointCloudAlpha"],
        markerStyle=".",
    )
    plt.axis("off")
    scale_axes_to_fit(ax, pointCloud, zoom=styleOpt_3D["zoom"])
    ax.view_init(elev=styleOpt_3D["elev"], azim=styleOpt_3D["azim"])
    # # customize figure
    # ax.axes.set_xlim3d(left=styleOpt_3D["axLimX"][0], right=styleOpt_3D["axLimX"][1])
    # ax.axes.set_ylim3d(bottom=styleOpt_3D["axLimY"][0], top=styleOpt_3D["axLimY"][1])
    # ax.axes.set_zlim3d(bottom=styleOpt_3D["axLimZ"][0], top=styleOpt_3D["axLimZ"][1])

    # save figure
    if controlOpt["save"]:
        fileName = (
            controlOpt["saveName"]
            + "_frame_"
            + str(frame)
            + "_3D"
            + controlOpt["saveFormat_3D"]
        )
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, method)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        filePath = os.path.join(folderPath, fileName)
        plt.savefig(filePath, dpi=controlOpt["dpi"], bbox_inches="tight", pad_inches=0)
        if controlOpt["verbose"]:
            print(
                "Saved 3D result for frame {} for method {} as {}".format(
                    frame, method, filePath
                )
            )
    # display figure
    if controlOpt["showResult_3D"]:
        plt.show(block=True)
    else:
        plt.close(fig)
    return None


def createResultPlot_Sim(dataSetResult, frame):
    registrationResult = eval.findCorrespondingEntryFromKeyValuePair(
        dataSetResult["trackingResults"]["kpr"]["registrations"], "frame", frame
    )
    pointCloud = registrationResult["Y"]
    modelParameters = dataSetResult["trackingResults"][method]["modelParameters"]

    bdloModel = eval.generateModel(modelParameters)
    q = registrationResult["q"]
    bdloModel.setBranchColorsFromColorPalette(styleOpt_Sim["colorPalette"])

    dartScene = eval.plotTrackingResultDartSim(
        q=q,
        bdloModel=bdloModel,
        camEye=styleOpt_Sim["camEye"],
        camCenter=styleOpt_Sim["camCenter"],
        camUp=styleOpt_Sim["camUp"],
    )
    dartScene.boardSkel.setColor(styleOpt_Sim["boardColor"])
    # setup image size
    dataSetPath = dataSetResult["dataSetPath"]
    rgbImg = eval.getDataSet(frame, dataSetPath)[0]
    imgShape = rgbImg.shape
    imgWidth = imgShape[1]
    imgHeight = imgShape[0]
    dartScene.setupSceneViewInWindow(x=0, y=0, width=imgWidth, height=imgHeight)

    if styleOpt_Sim["plotPointCloud"]:
        pointCloud = registrationResult["Y"]
        pointCloudSize = 0.005 if pointCloudSize is None else pointCloudSize
        pointCloudColor = [1, 0, 0] if pointCloudColor is None else pointCloudColor
        pointCloudAlpha = 1 if pointCloudAlpha is None else pointCloudAlpha
        dartScene.addPointCloud(
            points=pointCloud,
            colors=styleOpt_Sim["pointCloudColor"],
            alpha=styleOpt_Sim["pointCloudAlpha"],
        )

    if controlOpt["showResult_Sim"]:
        dartScene.viewer.run()

    # save simulation result
    if controlOpt["save"]:
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, method)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        saveFilePath = os.path.join(
            controlOpt["saveFolder"],
            dataSetName,
            "kpr",
            controlOpt["saveName"] + "_frame_" + str(frame) + "_Sim" + ".png",
        )
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
        dartScene.viewer.captureScreen(saveFilePath)
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
        if controlOpt["verbose"]:
            print(
                "Saved Simulation result for frame {} for method {} as {}".format(
                    frame, method, saveFilePath + ".png"
                )
            )
    return


if __name__ == "__main__":
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)
    # save inputs
    if controlOpt["saveInputs"]:
        for i, result in enumerate(results):
            for frame in controlOpt["frames"]:
                saveInputImage(result, frame)
    # create plot
    for i, result in enumerate(results):
        for method in controlOpt["methods"]:
            for frame in controlOpt["frames"]:
                if controlOpt["generateResults_2D"]:
                    createResultPlot_2D(result, frame, method)
                if controlOpt["generateResults_3D"]:
                    createResultPlot_3D(result, frame, method)
                if controlOpt["generateResults_Sim"] and method == "kpr":
                    createResultPlot_Sim(result, frame)
