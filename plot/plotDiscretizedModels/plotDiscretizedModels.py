import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.modelling.discretization import determineNumSegments
    from src.simulation.bdlo import BranchedDeformableLinearObject
    from src.modelling.topologyModel import topologyModel
    from src.sensing.dataHandler import DataHandler
    from src.visualization.colors import *
    from src.visualization.plotUtils import *
    from src.visualization.dartVisualizer import *
except:
    print("Imports for plotting discretized models failed.")
    raise

save = True
bendingRadius = 0.015
toleratedErrors = [0.03, 0.015, 0.005]

filePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120402_702175_image_rgb.png"
saveFolder = "imgs/discretization"
styleOpt = {"colorPalette": thesisColorPalettes["viridis"]}

if __name__ == "__main__":

    for i, toleratedError in enumerate(toleratedErrors):
        dataSetPath = os.path.dirname(os.path.dirname(filePath)) + "/"
        dataHandler = DataHandler(dataSetPath)

        # generate model
        modelParameters = dataHandler.getModelParameters()
        topology = topologyModel(modelParameters["adjacencyMatrix"])
        total_length = 0
        for numBranch, branch in enumerate(topology.getBranches()):
            total_length += topology.getBranchLength(numBranch)

        numSegments = determineNumSegments(
            total_length=total_length,
            minimal_bending_radius=bendingRadius,
            max_tolerated_error=toleratedErrors[i],
        )
        modelParameters["defaultNumBodyNodes"] = numSegments
        modelParameters["branchSpecs"][0]["radius"] = 0.007
        modelParameters["branchSpecs"][2]["radius"] = 0.007
        modelParameters["branchSpecs"][4]["radius"] = 0.007

        modelParameters["branchSpecs"][1]["rootJointRestPositions"][0] = 2.5
        modelParameters["branchSpecs"][3]["rootJointRestPositions"][0] = -2.7
        bdloModel = BranchedDeformableLinearObject(**modelParameters)

        # load model
        print(total_length)
        print(numSegments)
        bdloModel.setInitialPose(
            initialPosition=[0.4, 0.4, 0.3], initialRotation=[np.pi / 2, 0, np.pi / 2]
        )
        # bdloModel.setColor([0.5, 0.5, 0.5])
        # bdloModel.setBranchColorsFromColorPalette(styleOpt["colorPalette"])
        dartVis = DartScene(
            bdloModel.skel,
            q=bdloModel.getGeneralizedCoordinates(),
            loadRobot=True,
            loadCell=False,
            loadBoard=False,
        )
        height = 0.29
        dartVis.addBox(
            [0.5, 1.2, height], color=[0.3, 1, 0.5], offset=[0.4, 0, height / 2]
        )
        # bdloModel.setColor([0, 0, 0])
        eye = [1.2, 1.2, 1.7]
        center = np.mean(bdloModel.getCartesianBodyCenterPositions(), axis=0)
        up = [0, 0, 1]
        if save:
            dartVis.saveFrame(
                savePath=os.path.join(saveFolder, "discretizedModel_" + str(i)),
                eye=eye,
                center=center,
                up=up,
                width=896,
                height=1024,
            )
        else:
            dartVis.setCameraPosition(
                eye=eye,
                center=center,
                up=up,
            )
            dartVis.runVisualization()
    # load image
    rgbImage = cv2.imread(filePath)
    if save:
        cv2.imwrite(os.path.join(saveFolder, "realImg" + ".png"), rgbImage)
    else:
        cv2.imshow("RGB image", cv2.resize(rgbImage, None, fx=0.8, fy=0.8))
        cv2.waitKey(0)
    print("Done.")
