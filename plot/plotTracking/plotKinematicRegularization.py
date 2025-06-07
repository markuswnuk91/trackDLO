import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.lines as mlines
from scipy.stats import multivariate_normal

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
except:
    print("Imports for plotting kinematic regulatization for tracking chapter failed.")
    raise
runOpt = {
    "runInitialization": False,
    "runRegistrations": False,
    "saveInitializationResult": False,
    "saveRegistrationResults": False,
    "plotCorrespondances": True,
    "plotModelFitting": False,
    "plotKinematicRegularization": False,
}
visOpt = {"visualizeInitialLocalizationResult": False}
saveOpt = {
    "savePlots": True,
    "initializationResultPath": "data/plots/kinematicRegularization",
    "saveFolderPath": "imgs/kinematicRegularization",
    "dpi": 300,
}
styleOpt = {"branchColorMap": thesisColorPalettes["viridis"]}
n_th_element = 10
nodeSize = 30
edgeSize = 2
evalConfigPath = "plot/plotTracking/config.json"
# filePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120351_790481_image_rgb.png"
filePath_unoccluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120411_389179_image_rgb.png"
# "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140157_033099_image_rgb.png"
filePath_occluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120419_158610_image_rgb.png"
# "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140355_166626_image_rgb.png"


def determineCorrespondanceColors(model, Y, P=None):
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    branchColors = []
    for branchIndex in range(0, model.getNumBranches()):
        branchColor = styleOpt["branchColorMap"].to_rgba(
            branchIndex / (model.getNumBranches() - 1)
        )
        branchColors.append(branchColor[:3])
    branchColorMaps = []
    for i, branchColor in enumerate(branchColors):
        branchColor = np.array(branchColor + (1,))
        endColor = branchColor + 0.7 * (np.array([1, 1, 1, 1]) - branchColor)
        branchColorMap = LinearSegmentedColormap.from_list(
            "branchColorMap_i", [branchColor, endColor]
        )
        norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
        mapper = ScalarMappable(norm=norm, cmap=branchColorMap)
        branchColorMaps.append(mapper)
    X_colors = []
    for n in bodyNodeIndices:
        branchIndex = correspondingBranches[n]
        nodeIndicesInBranchList = model.getBranchBodyNodeIndices(branchIndex)
        nodeIndexInBranch = np.where(np.array(nodeIndicesInBranchList) == n)[0]
        nodeColor = branchColorMaps[branchIndex].to_rgba(
            nodeIndexInBranch[0] / len(nodeIndicesInBranchList)
        )
        X_colors.append(nodeColor)
    if P is not None:
        Y_colors = []
        for m, y in enumerate(Y):
            y_color = np.zeros(4)
            for n in bodyNodeIndices:
                y_color += P[n, m] * np.array(list(X_colors[n]))
            y_color += (1 - np.sum(P[:, m])) * np.array([0.5, 0.5, 0.5, 0.9])
            Y_colors.append(y_color)
        return X_colors, Y_colors
    else:
        return X_colors


def colorBranchesWithLinearColorMap(model, Y, P):
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    branchColorMap = styleOpt["branchColorMap"]
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    # branchColors = []
    # for branchIndex in range(0, model.getNumBranches()):
    #     branchColor = styleOpt["branchColorMap"].to_rgba(
    #         branchIndex / (model.getNumBranches() - 1)
    #     )
    #     branchColors.append(branchColor[:3])
    # branchColorMaps = []
    # for i, branchColor in enumerate(branchColors):
    #     branchColor = np.array(branchColor + (1,))
    #     endColor = branchColor + 0.7 * (np.array([1, 1, 1, 1]) - branchColor)
    #     branchColorMap = LinearSegmentedColormap.from_list(
    #         "branchColorMap_i", [branchColor, endColor]
    #     )
    #     norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
    #     mapper = ScalarMappable(norm=norm, cmap=branchColorMap)
    #     branchColorMaps.append(mapper)
    X_colors = []
    for n in bodyNodeIndices:
        branchIndex = correspondingBranches[n]
        nodeIndicesInBranchList = model.getBranchBodyNodeIndices(branchIndex)
        nodeIndexInBranch = np.where(np.array(nodeIndicesInBranchList) == n)[0]
        if model.isOuterBranch(model.getBranch(branchIndex)):
            if branchIndex != 0:
                nodeColor = branchColorMap.to_rgba(
                    nodeIndexInBranch[0] / (len(nodeIndicesInBranchList) - 1)
                )
            else:
                nodeColor = branchColorMap.to_rgba(
                    (len(nodeIndicesInBranchList) - nodeIndexInBranch[0])
                    / len(nodeIndicesInBranchList)
                )
        else:
            nodeColor = branchColorMap.to_rgba(0)
        X_colors.append(nodeColor)
    if P is not None:
        Y_colors = []
        for m, y in enumerate(Y):
            y_color = np.zeros(4)
            for n in bodyNodeIndices:
                y_color += P[n, m] * np.array(list(X_colors[n]))
            y_color += (1 - np.sum(P[:, m])) * np.array([0.5, 0.5, 0.5, 0.9])
            Y_colors.append(y_color)
        return np.array(X_colors)[:, :3], np.array(Y_colors)[:, :3]
    else:
        return np.array(X_colors)[:, :3]


if __name__ == "__main__":
    # load point set
    eval = TrackingEvaluation(evalConfigPath)
    fileName_unoccluded = os.path.basename(filePath_unoccluded)
    dataSetFolderPath = os.path.dirname(os.path.dirname(filePath_unoccluded)) + "/"
    (Y, _) = eval.getPointCloud(fileName_unoccluded, dataSetFolderPath)
    # get inital configuration
    if runOpt["runInitialization"]:
        initializationResult = eval.runInitialization(
            dataSetFolderPath, fileName_unoccluded
        )
        if runOpt["saveInitializationResult"]:
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
    modelParameters = initializationResult["modelParameters"]
    model = eval.generateModel(modelParameters)
    X_init = model.getCartesianBodyCenterPositions()
    if visOpt["visualizeInitialLocalizationResult"]:
        fig, ax = setupLatexPlot3D()
        plotPointSet(ax=ax, X=Y, color=[1, 0, 0], alpha=0.1, size=1)
        # mask = np.ones(len(Y), dtype=bool)
        # mask[300:1000] = False
        plt.show(block=False)
        plotPointSet(
            ax=ax,
            X=X_init,
            size=10,
            markerStyle="o",
            edgeColor=[0, 0, 1],
            color=[1, 1, 1],
        )
        scale_axes_to_fit(ax=ax, points=X_init)
        ax.view_init(elev=90, azim=0)
        plt.show(block=True)
    # generate plots
    # rotate model
    model.setInitialPose(initialRotation=[np.pi / 2, 0, np.pi / 2])
    # translate model
    X = model.getCartesianBodyCenterPositions()
    offset = np.mean(Y, axis=0) - np.mean(X, axis=0)
    initialPosition = X[0, :] + offset
    model.setInitialPose(initialPosition=initialPosition)
    # visualize correspondances
    if runOpt["plotCorrespondances"]:
        set_text_to_latex_font(scale_text=1)
        modelColor = [0, 0, 1]
        pointCloudColor = [1, 0, 0]
        X_init = initializationResult["localization"]["X"]
        X = model.getCartesianBodyCenterPositions()
        # initialCorrespondanceReg = CoherentPointDrift(
        #     **{"X": X, "Y": Y}, **eval.config["cpdParameters"]
        # )
        # P_init = initialCorrespondanceReg.estimateCorrespondance().copy()
        eval.config["cpdParameters"]["beta"] = 0.4
        eval.config["cpdParameters"]["alpha"] = 0.1
        eval.config["cpdParameters"]["sigma"] = None
        reg = CoherentPointDrift(**{"X": X, "Y": Y}, **eval.config["cpdParameters"])
        reg.register()
        X = reg.T
        initialCorrespondanceReg = CoherentPointDrift(
            **{"X": X, "Y": Y}, **eval.config["cpdParameters"]
        )
        # P = initialCorrespondanceReg.estimateCorrespondance().copy()
        P = reg.P
        X_colors, Y_colors = determineCorrespondanceColors(model, Y, P)
        # # #################################################################
        # # # plot input point set
        # # #################################################################
        # # get axis limits of previous plot to scale the same
        # fig, ax = setupLatexPlot3D()
        # plotPointSet(ax=ax, X=Y[::n_th_element], color=[1, 0, 0], size=1)
        # scale_axes_to_fit(ax=ax, points=Y[::n_th_element], zoom=1.5)
        # ax.view_init(elev=90, azim=0)
        # plt.axis("off")
        # plt.show(block=True)
        #################################################################
        # start plotting correspondances
        #################################################################
        fig_correspondances, ax = setupLatexPlot3D()
        X_offset = X + np.array([-0.0, 0, 0.5])
        Y_downsampled = Y[::n_th_element]
        plotPointSet(
            ax=ax,
            X=X_offset,
            color=[1, 1, 1],
            edgeColor=modelColor,
            size=20,
            lineWidth=1.5,
        )
        # plotBranchWiseColoredTopology3D(
        #     ax=ax,
        #     topology=model,
        #     colorPalette=thesisColorPalettes["viridis"],
        #     pointAlpha=0.0001,
        # )
        plotPointSet(
            ax=ax,
            X=Y[::n_th_element],
            color=pointCloudColor,
            size=1,
            alpha=0.9,
        )
        # n_highlight = list(range(0, model.getNumBodyNodes()))
        n_highlight = list(range(0, model.getNumBodyNodes()))
        # n_highlight = n_highlight[0::6]

        # for n in n_highlight:
        #     for m, y in enumerate(Y[::n_th_element]):
        #         # weight = P[:, ::n_th_element][n, m] / np.max(P[:, ::n_th_element][n, :])
        #         weight = P[:, ::n_th_element][n, m] / np.max(P[:, ::n_th_element][:, :])
        #         plotLine(
        #             ax=ax,
        #             pointPair=np.vstack((y, X[n, :])),
        #             color=thesisColorPalettes["viridis"].to_rgba(1.2 * weight),
        #             linewidth=0.5 + 2 * weight,
        #             alpha=0.1 + (0.8 * weight),
        #         )
        # colorMap = LinearSegmentedColormap.from_list(
        #     "colorMap", [[1, 1, 1], thesisColors["uniSLightBlue"]]
        # )
        # norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
        # colorPalette = ScalarMappable(norm=norm, cmap=colorMap)
        colorPalette = thesisColorPalettes["viridis"]
        for n in n_highlight:
            # get n largest elements
            largest_indices = np.argsort(P[:, ::n_th_element][n, :])[-40:]
            for m, y in enumerate(Y[::n_th_element][largest_indices, :]):
                # weight = P[:, ::n_th_element][n, m] / np.max(P[:, ::n_th_element][n, :])
                weight = P[:, ::n_th_element][n, largest_indices[m]] / np.max(
                    P[:, ::n_th_element][:, largest_indices]
                )
                plotLine(
                    ax=ax,
                    pointPair=np.vstack((y, X_offset[n, :])),
                    color=colorPalette.to_rgba(0.3 + 0.6 * weight),
                    # linewidth=0.5 + 1.5 * weight,
                    linewidth=1,
                    alpha=0.05 + (0.9 * weight),
                )
        plt.axis("off")
        scale_axes_to_fit(
            ax=ax,
            points=np.vstack((X_offset, Y[len(X_offset), :])),
            zoom=1.4,
        )
        axOffset = [0, 0, 0]
        ax.set_xlim(ax.get_xlim()[0] + axOffset[0], ax.get_xlim()[1] + axOffset[0])
        ax.set_ylim(ax.get_ylim()[0] + axOffset[1], ax.get_ylim()[1] + axOffset[1])
        ax.set_zlim(ax.get_zlim()[0] + axOffset[2], ax.get_zlim()[1] + axOffset[2])
        ax.view_init(elev=35, azim=40)
        # add ColorBar
        cbar = plt.colorbar(colorPalette, location="right", shrink=0.75)
        cbar.set_label("$p(\mathbf{x}_{b,n}^t \mid \mathbf{y}_{m}^t)$", rotation=270, labelpad=20)
        # Setting the color bar ticks and labels to reflect the bending radii
        # Convert bending radii to string labels if necessary for formatting
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["low", "high"])

        #################################################################
        # plot probability distribution
        #################################################################
        x_range = np.max(X[:, 0]) - np.min(X[:, 0])
        y_range = np.max(X[:, 1]) - np.min(X[:, 1])
        max_range = np.max([x_range, y_range])
        x_mid = 0.5 * (np.max(X[:, 0]) + np.min(X[:, 0]))
        y_mid = 0.5 * (np.max(X[:, 1]) + np.min(X[:, 1]))
        x_grid, y_grid = np.mgrid[
            x_mid - 0.5 * max_range : x_mid + 0.5 * max_range : 0.01,
            y_mid - 0.5 * max_range : y_mid + 0.5 * max_range : 0.01,
        ]
        xy_grid = np.column_stack([x_grid.flat, y_grid.flat])
        fig_probabilityDistribution = plt.figure()
        ax = fig_probabilityDistribution.add_subplot(111, projection="3d")
        z = np.zeros(x_grid.size)
        sigma2 = 1e-6
        for x in X[:, :2]:
            z += (len(X) * (2 * np.pi * sigma2) ** (len(x) / 2)) ** (
                -1
            ) * multivariate_normal.pdf(xy_grid, mean=x, cov=np.sqrt(sigma2))
        z = z.reshape(x_grid.shape)
        z = z / np.max(z)
        ax.plot_surface(x_grid, y_grid, z + 0.1, cmap="viridis", alpha=0.4)
        plotPointSet(
            ax=ax,
            X=np.hstack((X[:, :2], np.zeros((len(X), 1)))),
            markerStyle="o",
            color=[1, 1, 1],
            edgeColor=[0, 0, 1],
        )
        plotPointSet(
            ax=ax,
            X=np.hstack(
                (Y[::n_th_element][:, :2], np.zeros((len(Y[::n_th_element]), 1)))
            ),
            markerStyle=".",
            color=[1, 0, 0],
            alpha=0.5,
        )
        # set axis lables
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(r"probability $p(\mathbf{y}_m^t)$", rotation=90)
        # set background color as white
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=30, azim=40)
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=[1, 0, 0],
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=[0, 0, 1],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper right")
        # #################################################################
        # # plot input point set
        # #################################################################
        # # get axis limits of previous plot to scale the same
        # zoom = 1.4
        # x_lims = ax.get_xlim()
        # y_lims = ax.get_ylim()
        # z_lims = ax.get_zlim()
        # # level point cloud to zero
        # Yzero = Y[::n_th_element]
        # Yzero[:, 2] = 0
        # fig_inputPointCloud, ax = setupLatexPlot3D()
        # plotPointSet(ax=ax, X=Yzero, color=[1, 0, 0], size=1)
        # # scale_axes_to_fit(ax=ax, points=Y[::n_th_element])
        # ax.view_init(elev=40, azim=40)
        # axOffset = [0.1, 0.1, 0]
        # ax.set_xlim((x_lims[0] / zoom + axOffset[0], x_lims[1] / zoom + axOffset[0]))
        # ax.set_ylim((y_lims[0] / zoom + axOffset[1], y_lims[1] / zoom + axOffset[1]))
        # ax.set_zlim((z_lims[0] / zoom + axOffset[2], z_lims[1] / zoom + axOffset[2]))
        # # x_lims = ax.get_xlim()
        # # y_lims = ax.get_ylim()
        # # z_lims = ax.get_zlim()
        # # z_range = z_lims[1] - z_lims[0]
        # # z_min = np.mean(Y, axis=0)[2]
        # # ax.set_zlim(z_min, z_min + z_range)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        if saveOpt["savePlots"]:
            # fig_inputPointCloud.savefig(
            #     os.path.join(saveOpt["saveFolderPath"], "InputPointCloud"),
            #     bbox_inches="tight",
            #     pad_inches=0.4,
            #     dpi=saveOpt["dpi"],
            # )
            fig_probabilityDistribution.savefig(
                os.path.join(saveOpt["saveFolderPath"], "GMMDistribution.pdf"),
                bbox_inches="tight",
                pad_inches=0.5,
                dpi=saveOpt["dpi"],
            )
            fig_correspondances.savefig(
                os.path.join(
                    saveOpt["saveFolderPath"],
                    "CorrespondenceVisualization.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0.4,
                # dpi=saveOpt["dpi"],
            )
        plt.show(block=True)

    ###################################################################
    # plot EM Update
    ###################################################################
    if runOpt["plotModelFitting"]:
        widthHeightRatio = 2
        n_th_element = 10
        X = model.getCartesianBodyCenterPositions()
        X_colors = determineCorrespondanceColors(model)
        fig_fitting_1, ax = setupLatexPlot3D()
        set_text_to_latex_font(scale_text=1.2)
        plotPointSet(
            ax=ax,
            X=X,
            color=[1, 1, 1],
            edgeColor=X_colors,
            size=nodeSize,
            lineWidth=edgeSize,
        )
        plotPointSet(
            ax=ax,
            X=Y[::n_th_element],
            color=[1, 0, 0],
            size=1,
            alpha=0.9,
        )
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=[1, 0, 0],
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=X_colors[int(len(X_colors) / 2)],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper left")
        plt.axis("off")
        scale_axes_to_fit(ax=ax, points=X, zoom=1.5)
        ax.view_init(
            elev=90,
            azim=0,
        )
        ######################################################
        fig_fitting_2, ax = setupLatexPlot3D(widthHeightRatio=widthHeightRatio)
        eval.config["cpdParameters"]["sigma"] = 0.001
        initialCorrespondanceReg = CoherentPointDrift(
            **{"X": X, "Y": Y}, **eval.config["cpdParameters"]
        )
        P_init = initialCorrespondanceReg.estimateCorrespondance().copy()
        initialCorrespondanceReg.update_variance()
        print("Likelihood: {}".format(initialCorrespondanceReg.L))
        X_colors, Y_colors = determineCorrespondanceColors(model, Y, P_init)

        plotPointSet(
            ax=ax,
            X=X,
            color=[1, 1, 1],
            edgeColor=X_colors,
            size=nodeSize,
            lineWidth=edgeSize,
        )
        plotPointSet(
            ax=ax,
            X=Y[::n_th_element],
            color=np.array(Y_colors)[:, :3][::n_th_element],
            size=1,
            alpha=0.9,
        )
        plt.axis("off")
        scale_axes_to_fit(ax=ax, points=X, zoom=1.5)
        ax.view_init(
            elev=90,
            azim=0,
        )
        ######################################################
        fig_fitting_3, ax = setupLatexPlot3D(widthHeightRatio=widthHeightRatio)
        eval.config["cpdParameters"]["beta"] = 0.4
        eval.config["cpdParameters"]["alpha"] = 0.1
        eval.config["cpdParameters"]["sigma"] = None
        correspondanceRegistration = CoherentPointDrift(
            **{"X": X, "Y": Y}, **eval.config["cpdParameters"]
        )
        # correspondanceRegistration.registerCallback(
        #     eval.getVisualizationCallback(correspondanceRegistration)
        # )
        correspondanceRegistration.register()
        print("Likelihood: {}".format(correspondanceRegistration.L))
        X_colors, Y_colors = determineCorrespondanceColors(
            model, Y, correspondanceRegistration.P.copy()
        )
        plotPointSet(
            ax=ax,
            X=correspondanceRegistration.T,
            color=[1, 1, 1],
            size=nodeSize,
            edgeColor=X_colors,
            lineWidth=edgeSize,
        )
        plotPointSet(
            ax=ax,
            X=Y[::n_th_element],
            color=Y_colors[::n_th_element],
            size=1,
            alpha=0.9,
        )
        plt.axis("off")
        offset = np.array([0, 0.1, 0])
        scale_axes_to_fit(ax=ax, points=X + offset, zoom=1.5)
        ax.view_init(
            elev=90,
            azim=0,
        )
        if saveOpt["savePlots"]:
            fig_fitting_1.savefig(
                os.path.join(saveOpt["saveFolderPath"], "EMUpdate_Init.pdf"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=saveOpt["dpi"],
            )
            fig_fitting_2.savefig(
                os.path.join(saveOpt["saveFolderPath"], "EMUpdate_P-step.pdf"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=saveOpt["dpi"],
            )
            fig_fitting_3.savefig(
                os.path.join(saveOpt["saveFolderPath"], "EM_Update_M-step.pdf"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=saveOpt["dpi"],
            )
        plt.show(block=True)

    ###################################################################
    # plot Kinematic Regularization
    ###################################################################
    if runOpt["plotKinematicRegularization"]:
        n_th_element = 5
        pointCloudColor = [1, 0, 0]
        pointCloudAlpha = 0.3
        nodeColor = [0, 0, 1]
        lineWidth = 2
        offset = np.array([-0.1, 0.0, 0])
        set_text_to_latex_font(scale_text=1.5)
        oldConfigurationAlpha = 0.1
        X_init = initializationResult["localization"]["X"]
        q_init = initializationResult["localization"]["q"]
        model.setGeneralizedCoordinates(q_init)
        kinematicModel = KinematicsModelDart(model.skel.clone())
        zoom = 1.5
        # load occluded point set
        fileName_occluded = os.path.basename(filePath_occluded)
        (Y_occluded, _) = eval.getPointCloud(fileName_occluded, dataSetFolderPath)
        mask = np.ones(len(Y_occluded), dtype=bool)
        # mask[50:300] = False
        mask[1100:1400] = False
        Y_occluded = Y_occluded[mask]
        X_pre = model.getCartesianBodyCenterPositions()
        Y_pre = Y[::n_th_element]
        if runOpt["runRegistrations"]:
            # run tracking (GMM)
            eval.config["cpdParameters"][
                "alpha"
            ] = 0.01  # set low to make irrelevant such that CPD = GMM | zero not possible due to singularity
            eval.config["cpdParameters"]["beta"] = 0.1
            eval.config["cpdParameters"]["sigma"] = None
            cpd = CoherentPointDrift(
                **{"X": X_pre, "Y": Y_occluded}, **eval.config["cpdParameters"]
            )
            P = cpd.estimateCorrespondance()
            cpd.registerCallback(eval.getVisualizationCallback(cpd))
            gmmResult = eval.runRegistration(registration=cpd, checkConvergence=False)
            # cpd.register()

            # eval.config["sprParameters"]["alpha"] = 0.01
            # eval.config["sprParameters"]["beta"] = 0.1
            # eval.config["sprParameters"]["sigma"] = None
            spr = StructurePreservedRegistration(
                **{"X": X_pre, "Y": Y_occluded}, **eval.config["sprParameters"]
            )
            spr.registerCallback(eval.getVisualizationCallback(spr))
            # spr.register()
            sprResult = eval.runRegistration(registration=spr, checkConvergence=False)

            # run tracking Regularized
            eval.config["kprParameters"]["sigma2"] = None
            kpr = KinematicsPreservingRegistration(
                Y=Y_occluded,
                qInit=q_init,
                model=kinematicModel,
                **eval.config["kprParameters"]
            )
            kpr.registerCallback(eval.getVisualizationCallback(kpr))
            # kpr.register()
            kprResult = eval.runRegistration(registration=kpr, checkConvergence=False)
            if runOpt["saveRegistrationResults"]:
                eval.saveWithPickle(
                    data=gmmResult,
                    filePath=os.path.join(
                        saveOpt["initializationResultPath"],
                        "gmmResult.pkl",
                    ),
                    recursionLimit=10000,
                )
                eval.saveWithPickle(
                    data=sprResult,
                    filePath=os.path.join(
                        saveOpt["initializationResultPath"],
                        "sprResult.pkl",
                    ),
                    recursionLimit=10000,
                )
                eval.saveWithPickle(
                    data=kprResult,
                    filePath=os.path.join(
                        saveOpt["initializationResultPath"],
                        "kprResult.pkl",
                    ),
                    recursionLimit=10000,
                )
        else:
            # load registration results from saved files
            try:
                # load reesults from files
                gmmResult = eval.loadResults(
                    os.path.join(
                        saveOpt["initializationResultPath"],
                        "gmmResult.pkl",
                    )
                )
                sprResult = eval.loadResults(
                    os.path.join(
                        saveOpt["initializationResultPath"],
                        "sprResult.pkl",
                    )
                )
                kprResult = eval.loadResults(
                    os.path.join(
                        saveOpt["initializationResultPath"],
                        "kprResult.pkl",
                    )
                )
            except:
                print(
                    "Could not load registrations results. Make sure there are registrations results in : {}".format(
                        saveOpt["initializationResultPath"]
                    )
                )
        # save correspondance estimation results
        initializationResult = eval.loadResults(
            os.path.join(
                saveOpt["initializationResultPath"],
                "initializationResult.pkl",
            )
        )
        # plot initial configuration
        reg = NonRigidRegistration(**{"X": X_pre, "Y": Y_pre})
        reg.sigma2 = 0.001
        fig_regularization_init, ax = setupLatexPlot3D()
        # X_colors, Y_colors = determineCorrespondanceColors(
        #     model, Y_pre, P=reg.estimateCorrespondance()
        # )
        X_colors, Y_colors = colorBranchesWithLinearColorMap(
            model, Y_pre, reg.estimateCorrespondance().copy()
        )
        plotPointSet(
            ax=ax,
            X=X_pre,
            color=[1, 1, 1],
            size=nodeSize,
            edgeColor=X_colors,
            lineWidth=lineWidth,
        )
        plotPointSet(
            ax=ax, X=Y_pre, color=pointCloudColor, size=1, alpha=pointCloudAlpha
        )
        plotGraph3D(
            ax=ax,
            X=X_pre,
            adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
            pointAlpha=0,
            lineAlpha=0.5,
            lineStyle="--",
        )
        scale_axes_to_fit(ax=ax, points=Y_pre + offset, zoom=zoom)
        ax.view_init(elev=90, azim=0)
        plt.axis("off")
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=pointCloudColor,
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud $\mathbf{Y}^t$",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=styleOpt["branchColorMap"].to_rgba(0)[:3],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids  $\mathbf{X}^t$",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper left")

        # plot gmm result
        fig_regularization_gmm, ax = setupLatexPlot3D()
        # plotPointSet(
        #     ax=ax, X=gmmResult["T"], size=30, color=[1, 1, 1], edgeColor=[0, 0, 1]
        # )
        # plotGraph3D(
        #     ax=ax,
        #     X=gmmResult["T"],
        #     adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
        # )
        reg.T = gmmResult["T"]
        reg.Y = gmmResult["Y"]
        reg.sigma2 = gmmResult["sigma2"]
        X_colors, Y_colors = colorBranchesWithLinearColorMap(
            model, Y_occluded, reg.estimateCorrespondance().copy()
        )
        # determineCorrespondanceColors(
        #     model, Y_occluded, P=reg.estimateCorrespondance().copy()
        # )
        plotPointSet(
            ax=ax,
            X=gmmResult["T"],
            color=[1, 1, 1],
            edgeColor=X_colors,
            lineWidth=lineWidth,
        )
        plotPointSet(
            ax=ax,
            X=Y_occluded[::n_th_element],
            size=1,
            color=pointCloudColor,
            alpha=pointCloudAlpha,
        )
        plotGraph3D(
            ax=ax,
            X=gmmResult["T"],
            adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
            pointAlpha=0,
            lineAlpha=0.5,
            lineStyle="--",
        )
        ax.view_init(elev=90, azim=0)
        plt.axis("off")
        scale_axes_to_fit(ax=ax, points=Y_pre + offset, zoom=zoom)
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=pointCloudColor,
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud $\mathbf{Y}^{t+1}$",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=styleOpt["branchColorMap"].to_rgba(0)[:3],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids  $\mathbf{X}^{t+1}$",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper left")

        # plot spr result
        fig_regularization_spr, ax = setupLatexPlot3D()
        reg.T = sprResult["T"]
        reg.Y = sprResult["Y"]
        reg.sigma2 = sprResult["sigma2"]
        # X_colors, Y_colors = determineCorrespondanceColors(
        #     model, Y_occluded, P=reg.estimateCorrespondance().copy()
        # )
        X_colors, Y_colors = colorBranchesWithLinearColorMap(
            model, Y_occluded, sprResult["P"]
        )
        # plotPointSet(
        #     ax=ax, X=sprResult["X"], color=[1, 1, 1], edgeColor=[0, 0, 1], alpha=0.3
        # )
        plotPointSet(
            ax=ax,
            X=sprResult["T"],
            color=[1, 1, 1],
            edgeColor=X_colors,
            lineWidth=lineWidth,
        )
        plotPointSet(
            ax=ax,
            X=Y_occluded[::n_th_element],
            size=1,
            color=pointCloudColor,
            alpha=pointCloudAlpha,
        )
        # plotCorrespondancesAsArrows3D(
        #     ax=ax,
        #     X=sprResult["X"],
        #     Y=sprResult["T"],
        #     alpha=oldConfigurationAlpha,
        # )
        # plotPointSet(
        #     ax=ax,
        #     X=sprResult["T"],
        #     size=30,
        #     color=[1, 1, 1],
        #     edgeColor=[0, 0, 1],
        #     lineWidth=2,
        # )
        plotGraph3D(
            ax=ax,
            X=sprResult["T"],
            adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
            pointAlpha=0,
            lineAlpha=0.5,
            lineStyle="--",
        )
        # plotPointSet(ax=ax, X=Y_occluded, size=1, color=[1, 0, 0])
        scale_axes_to_fit(ax=ax, points=Y_occluded + offset, zoom=zoom)
        ax.view_init(elev=90, azim=0)
        plt.axis("off")
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=pointCloudColor,
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud $\mathbf{Y}^{t+1}$",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=styleOpt["branchColorMap"].to_rgba(0)[:3],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids  $\mathbf{X}^{t+1}$",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper left")

        # plot kpr result
        fig_regularization_kpr, ax = setupLatexPlot3D()
        reg.T = kprResult["T"]
        reg.Y = kprResult["Y"]
        reg.sigma2 = kprResult["sigma2"]
        P_kpr = reg.estimateCorrespondance().copy()
        # X_colors, Y_colors = determineCorrespondanceColors(
        #     model, Y_occluded, P=kprResult["P"]
        # )
        X_colors, Y_colors = colorBranchesWithLinearColorMap(
            model, Y_occluded, kprResult["P"]
        )
        plotPointSet(
            ax=ax,
            X=kprResult["T"],
            color=[1, 1, 1],
            edgeColor=X_colors,
            lineWidth=lineWidth,
        )
        plotPointSet(
            ax=ax,
            X=Y_occluded[::n_th_element],
            size=1,
            color=pointCloudColor,
            alpha=pointCloudAlpha,
        )
        plotGraph3D(
            ax=ax,
            X=kprResult["T"],
            adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
            pointAlpha=0,
            lineAlpha=0.5,
            lineStyle="--",
        )
        # plotPointSet(
        #     ax=ax, X=kprResult["T"], size=30, color=[1, 1, 1], edgeColor=[0, 0, 1]
        # )
        # plotPointSet(ax=ax, X=Y_occluded, size=1, color=[1, 0, 0])
        # plotGraph3D(
        #     ax=ax,
        #     X=kprResult["T"],
        #     adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix(),
        # )
        scale_axes_to_fit(ax=ax, points=Y_occluded + offset, zoom=zoom)
        ax.view_init(elev=90, azim=0)
        plt.axis("off")
        # create legend
        legendSymbols = []
        pointCloudMarker = mlines.Line2D(
            [],
            [],
            color=pointCloudColor,
            marker=".",
            linestyle="None",
            markersize=1,
            label=r"Point cloud $\mathbf{Y}^{t+1}$",
        )
        gaussianCentroidMarker = mlines.Line2D(
            [],
            [],
            color=[1, 1, 1],
            markeredgecolor=styleOpt["branchColorMap"].to_rgba(0)[:3],
            marker="o",
            linestyle="None",
            label=r"Gaussian centroids  $\mathbf{X}^{t+1}$",
        )
        legendSymbols.append(pointCloudMarker)
        legendSymbols.append(gaussianCentroidMarker)
        ax.legend(handles=legendSymbols, loc="upper left")

        if saveOpt["savePlots"]:
            fig_regularization_init.savefig(
                os.path.join(
                    saveOpt["saveFolderPath"], "kinematicRegularization_Init.pdf"
                ),
                bbox_inches="tight",
                pad_inches=0,
                dpi=saveOpt["dpi"],
            )
            fig_regularization_gmm.savefig(
                os.path.join(
                    saveOpt["saveFolderPath"],
                    "kinematicRegularization_GMM.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0,
                # dpi=saveOpt["dpi"],
            )
            fig_regularization_spr.savefig(
                os.path.join(
                    saveOpt["saveFolderPath"],
                    "kinematicRegularization_SPR.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0,
                # dpi=saveOpt["dpi"],
            )
            fig_regularization_kpr.savefig(
                os.path.join(
                    saveOpt["saveFolderPath"],
                    "kinematicRegularization_KPR.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0,
                # dpi=saveOpt["dpi"],
            )
        plt.show(block=True)
    print("Done.")
