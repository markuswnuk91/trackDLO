import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import norm, gamma
import tikzplotlib
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from scipy.stats import chi2

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.visualization.plot2D import *
    from src.visualization.colors import *
    from src.visualization.plotUtils import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = GraspingAccuracyEvaluation()

controlOpt = {
    "resultsToLoad": [-1],
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],
    "showPlot": True,
    "makeScatterPlot": True,
    "makeOutlierRatioBarPlot": True,
    "makeHistogramPlot": True,
    "makeGraspingAccuracyBoxPlot": True,
    "save": True,
    "saveAs": "PDF",  # "PDF", "TIKZ", "PGF"
    # "saveAsTikz": False,  # does not work with dashed lines, and does not plot legend
    # "saveAsPGF": False,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/graspingAccuracyEvaluationResults",
    "saveFileNameScatterPlot": "graspingErrorsScatterPlot",
    "saveFileNameOutlierRatio": "successRateBarPlot",
    "saveFileNameHistogramTranslational": "translationalGraspingErrorsHistogram",
    "saveFileNameHistogramRotaional": "rotationalGraspingErrorsHistogram",
    "saveFileNameBoxPlotTranslational": "translationalGraspingAccuracyBoxPlot",
    "saveFileNameBoxPlotRotational": "rotationalGraspingAccuracyBoxPlot",
}

styleOpt = {
    "figureWidth": 7.4,
    "figureHeight": 5.8,
    "methodColors": {
        "cpd": thesisColorPalettes["viridis"].to_rgba(0)[:3],
        "spr": thesisColorPalettes["viridis"].to_rgba(0.5)[:3],
        "kpr": thesisColorPalettes["viridis"].to_rgba(1)[:3],
    },
    "modelMarkers": {
        "modelY": "o",
        "partial": "o",  # "s"
        "arena": "o",  # "^"
    },
    "alpha": 0.7,
    "markersize": 20,
    "legendMarkerSize": 5,
    # scatterPlot
    "plotTreshold": "gaussian",  # "lines", "gaussian_fit"
    "translationalErrorThreshold": 0.05,  # None: do not plot
    "rotationalErrorThreshold": 45,  # None: do not plot
    "translationalThresholdLineColor": [1, 0, 0],
    "rotationalThresholdLineColor": [1, 0, 0],
    "thresholdLineStyle": "--",
    "gaussianThresholdContourColorpalette": "Reds",
    "gaussianThresholdContourAlpha": 0.5,
    "gaussianThresholdLineColor": "blue",
    "gaussianThresholdCenterColor": [1, 0, 0],
    "gaussianThresholdCenterAlpha": 0.5,
    "gaussianInlierDistributionAlpha": 0.1,
    "gaussianOutlierDistributionAlpha": 0.1,
    "gaussianInlierDistributionColor": "blue",
    "gaussianOutlierDistributionColor": "red",
    "gaussianSigmaFactor": 2,
    "histogramFitLineStyle": "--",
    # histogram plot
    "addSecondaryAxis": False,
    # successRateBarPlot
    "plotSuccessRateText": True,
    "plotSuccessRateLegend": False,
    # box plot
    "boxPlotFaceAlpha": 0.9,
}
# textwidth_in_pt = 483.6969
# figureScaling = 1
# latexFontSize_in_pt = 14
# latexFootNoteFontSize_in_pt = 10
# desiredFigureWidth = figureScaling * textwidth_in_pt
# desiredFigureHeight = figureScaling * textwidth_in_pt
# tex_fonts = {
#     #    "pgf.texsystem": "pdflatex",
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": latexFootNoteFontSize_in_pt,
#     "font.size": latexFootNoteFontSize_in_pt,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": latexFootNoteFontSize_in_pt,
#     "xtick.labelsize": latexFootNoteFontSize_in_pt,
#     "ytick.labelsize": latexFootNoteFontSize_in_pt,
# }
# figure font configuration
latexFontSize_in_pt = 20
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFontSize_in_pt,
    "xtick.labelsize": latexFontSize_in_pt,
    "ytick.labelsize": latexFontSize_in_pt,
}

# if controlOpt["saveAs"] == "PDF":
#     # figure font configuration
#     latexFontSize_in_pt = 16
#     tex_fonts = {
#         #    "pgf.texsystem": "pdflatex",
#         # Use LaTeX to write all text
#         "text.usetex": True,
#         "font.family": "serif",
#         # Use 10pt font in plots, to match 10pt font in document
#         "axes.labelsize": latexFontSize_in_pt,
#         "font.size": latexFontSize_in_pt,
#         # Make the legend/label fonts a little smaller
#         "legend.fontsize": latexFontSize_in_pt,
#         "xtick.labelsize": latexFontSize_in_pt,
#         "ytick.labelsize": latexFontSize_in_pt,
#     }
plt.rcParams.update(tex_fonts)

if controlOpt["saveAs"] == "PGF":
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    matplotlib.rcParams.update(tex_fonts)


resultFileName = "result.pkl"
resultFolderPaths = [
    "data/eval/graspingAccuracy/results/20230522_130903_modelY",
    "data/eval/graspingAccuracy/results/20230522_131545_modelY",
    "data/eval/graspingAccuracy/results/20230522_154903_modelY",
    "data/eval/graspingAccuracy/results/20230807_142319_partial",
    "data/eval/graspingAccuracy/results/20230807_142909_partial",
    "data/eval/graspingAccuracy/results/20230807_143737_partial",
    "data/eval/graspingAccuracy/results/20230522_140014_arena",
    "data/eval/graspingAccuracy/results/20230522_141025_arena",
    "data/eval/graspingAccuracy/results/20230522_142058_arena",
]


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def evaluateGraspingAccuracy(
    result,
    method,
    num,
):
    registrationResult = result["trackingResults"][method]["registrationResults"][num]
    frame = registrationResult["frame"]
    dataSetPath = result["dataSetPath"]

    # ground truth
    (
        groundTruthGraspingPose,
        groundTruthGraspingPosition,
        groundTruthGraspingRotationMatrix,
    ) = eval.loadGroundTruthGraspingPose(
        dataSetPath, frame + 1
    )  # ground truth grasping position is given by the frame after the prediction frame
    groundTruthGraspingAxis = groundTruthGraspingRotationMatrix[:3, 0]
    # prediction
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)
    graspingLocalCoordinate = graspingLocalCoordinates[num]
    T = registrationResult["result"]["T"]
    B = result["trackingResults"][method]["B"]
    S = result["initializationResult"]["localizationResult"]["SInit"]
    (
        predictedGraspingPosition,
        predictedGraspingAxis,
    ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
        T, B, S, graspingLocalCoordinate
    )
    graspingAccuracy = eval.calculateGraspingAccuracyError(
        predictedGraspingPosition=predictedGraspingPosition,
        predictedGraspingAxis=predictedGraspingAxis,
        groundTruthGraspingPose=groundTruthGraspingPose,
    )
    return graspingAccuracy


def scatterPlotGraspingErrors(
    translationalGraspingErrors,
    rotationalGraspingErrors,
    correspondingMethods,
    correspondingModelNames,
    colors=None,
    markers=None,
    alpha=0.3,
    translationalThreshold=None,
    rotationalThreshold=None,
    translationalThresholdLineColor=None,
    rotationalThresholdLineColor=None,
    thresholdLineStyle=None,
):
    # fig, ax = setupLatexPlot2D(
    #     figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    # )
    fig = plt.figure(figsize=(styleOpt["figureWidth"], styleOpt["figureHeight"]))
    ax = fig.add_subplot()

    if colors is None:
        colors = []
        for method in correspondingMethods:
            if method == "cpd":
                colors.append([1, 0, 0, alpha])
            elif method == "spr":
                colors.append([0, 0, 1, alpha])
            elif method == "kpr":
                colors.append([0, 1, 0, alpha])
            elif method == "krcpd":
                colors.append([1, 1, 0, alpha])
            else:
                colors.append([0.7, 0.7, 0.7, alpha])
    if markers is None:
        markers = []
        for model in correspondingModelNames:
            if model == "modelY":
                markers.append("o")
            elif model == "partial":
                markers.append("^")
            elif model == "arena":
                markers.append("s")
            elif model == "singleDLO":
                markers.append("D")

    for i, (transplationalError, rotationalError) in enumerate(
        zip(translationalGraspingErrors, rotationalGraspingErrors)
    ):
        ax.scatter(
            transplationalError,
            rotationalError,
            color=colors[i],
            marker=markers[i],
            alpha=alpha,
            s=styleOpt["markersize"],
        )
    # plot threshold lines
    if styleOpt["plotTreshold"] == "lines":
        translationalThresholdLineColor = (
            [1, 0, 0]
            if translationalThresholdLineColor is None
            else translationalThresholdLineColor
        )
        rotationalThresholdLineColor = (
            [1, 0, 0]
            if rotationalThresholdLineColor is None
            else rotationalThresholdLineColor
        )
        thresholdLineStyle = "-" if thresholdLineStyle is None else thresholdLineStyle
        if translationalThreshold is not None and rotationalThreshold is not None:
            plt.axvline(
                x=translationalThreshold,
                ymin=0,
                ymax=(rotationalThreshold - ax.get_ylim()[0])
                / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                color=translationalThresholdLineColor,
                linestyle=thresholdLineStyle,
            )
            plt.axhline(
                y=rotationalThreshold,
                xmin=0,
                xmax=(translationalThreshold - ax.get_xlim()[0])
                / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                color=rotationalThresholdLineColor,
                linestyle=thresholdLineStyle,
            )
    elif styleOpt["plotTreshold"] == "gaussian":
        # plot thenshold lines by gaussian fit
        # fit multivariat gaussian
        data = np.stack(
            (np.array(translationalGraspingErrors), np.array(rotationalGraspingErrors)),
            axis=1,
        )
        gm = GaussianMixture(n_components=2, covariance_type="full")
        gm.fit(data)

        # plot gaussians
        x_grid = np.linspace(0, np.max(translationalGraspingErrors))
        y_grid = np.linspace(0, np.max(rotationalGraspingErrors))
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
        # Z_inlier =
        # Z = -gm.score_samples(XX)
        # Z = Z.reshape(X_grid.shape)
        # CS = plt.contour(
        #     X_grid,
        #     Y_grid,
        #     Z,
        #     norm=LogNorm(vmin=1.0, vmax=1000.0),
        #     levels=np.logspace(0, 3, 10),
        #     cmap=styleOpt["gaussianThresholdContourColorpalette"],
        #     alpha=styleOpt["gaussianThresholdContourAlpha"],
        # )
        # Separate inlier and outlier Gaussians based on mean and variance (covariance)
        means = gm.means_
        covariances = gm.covariances_

        # Calculate the determinant of the covariance to find the one with the larger spread (outlier)
        determinants = np.array([np.linalg.det(cov) for cov in covariances])

        # Find the indices of the inlier and outlier Gaussians
        inlier_idx = np.argmin(determinants)  # Inlier has lower variance
        outlier_idx = np.argmax(determinants)  # Outlier has higher variance

        # Generate the probability densities for each Gaussian component using score_samples
        # Inlier Gaussian
        gm_inlier = GaussianMixture(n_components=1, covariance_type="full")
        gm_inlier.means_ = means[inlier_idx].reshape(1, -1)
        gm_inlier.covariances_ = covariances[inlier_idx].reshape(1, 2, 2)
        gm_inlier.weights_ = np.array([1.0])
        gm_inlier.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(covariances[inlier_idx])
        ).reshape(1, 2, 2)

        Z_inlier = -gm_inlier.score_samples(XX).reshape(X_grid.shape)

        # Outlier Gaussian
        gm_outlier = GaussianMixture(n_components=1, covariance_type="full")
        gm_outlier.means_ = means[outlier_idx].reshape(1, -1)
        gm_outlier.covariances_ = covariances[outlier_idx].reshape(1, 2, 2)
        gm_outlier.weights_ = np.array([1.0])
        gm_outlier.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(covariances[outlier_idx])
        ).reshape(1, 2, 2)

        Z_outlier = -gm_outlier.score_samples(XX).reshape(X_grid.shape)

        # Plotting inlier Gaussian (blue)
        inlier_levels = np.logspace(0, 1, 5)  # 5 equidistant levels
        plt.contour(
            X_grid,
            Y_grid,
            Z_inlier,
            levels=inlier_levels,
            colors=styleOpt["gaussianInlierDistributionColor"],
            linewidths=2,
            alpha=styleOpt[
                "gaussianOutlierDistributionAlpha"
            ],  # Adjust alpha for better visibility
        )

        # Plotting outlier Gaussian (red)
        outlier_levels = np.logspace(0, 1, 5)  # 5 equidistant levels
        plt.contour(
            X_grid,
            Y_grid,
            Z_outlier,
            levels=outlier_levels,
            colors=styleOpt["gaussianOutlierDistributionColor"],
            linewidths=2,
            alpha=styleOpt[
                "gaussianInlierDistributionAlpha"
            ],  # Adjust alpha for better visibility
        )

        # Highlight the means
        plt.scatter(
            means[inlier_idx][0],
            means[inlier_idx][1],
            color="blue",
            marker="o",
            s=100,
            label="Inlier Mean",
            alpha=styleOpt["gaussianThresholdCenterAlpha"],
        )
        plt.scatter(
            means[outlier_idx][0],
            means[outlier_idx][1],
            color="red",
            marker="o",
            s=100,
            label="Outlier Mean",
            alpha=styleOpt["gaussianThresholdCenterAlpha"],
        )

        # Set the factor of the standard deviation (1, 2, or any value you want)
        factor = styleOpt[
            "gaussianSigmaFactor"
        ]  # You can change this to any factor you want, e.g., 1 for 1-sigma, 3 for 3-sigma, etc.

        # Calculate the Mahalanobis distance for the specified factor of standard deviations
        # For 2D Gaussian, the critical value at a given factor corresponds to the chi-squared distribution
        d_sigma = np.sqrt(
            chi2.ppf(chi2.cdf(factor**2, df=2), df=2)
        )  # Mahalanobis distance for desired factor
        level_sigma = d_sigma**2  # Mahalanobis distance squared

        # Highlight the contour for the specified standard deviation factor
        plt.contour(
            X_grid,
            Y_grid,
            Z_inlier,
            levels=[level_sigma],
            colors=styleOpt["gaussianThresholdLineColor"],
            linewidths=2,
            linestyles=styleOpt["thresholdLineStyle"],
        )
        # # Highlight the mean with a point
        # mean = gm.means_[0]
        # plt.scatter(
        #     mean[0],
        #     mean[1],
        #     color=styleOpt["gaussianThresholdCenterColor"],
        #     marker="o",
        #     s=100,
        #     # label="Gaussian Mean",
        #     alpha=styleOpt["gaussianThresholdCenterAlpha"],
        # )
        # create legend
        methodsToList = list(set(correspondingMethods))
        # Define the custom sort order
        custom_order = ["cpd", "spr", "kpr"]
        # Sort the list based on the custom order
        methodsToList = sorted(methodsToList, key=lambda x: custom_order.index(x))
        legendSymbols = []
        for label in methodsToList:
            legendSymbol = Line2D(
                [],
                [],
                marker=markers[correspondingMethods.index(label)],
                color=colors[correspondingMethods.index(label)],
                linestyle="None",
                label=label.upper(),
                markersize=styleOpt["legendMarkerSize"],
            )
            legendSymbols.append(legendSymbol)

        if styleOpt["plotTreshold"] == "gaussian":
            gaussianInlierDistributionSymbol = configureLegendSymbol(
                style="line", color=styleOpt["gaussianInlierDistributionColor"]
            )
            gaussianInlierDistributionSymbol.set_label("inlier dist.")
            legendSymbols.append(gaussianInlierDistributionSymbol)
            gaussianOutlierDistributionSymbol = configureLegendSymbol(
                style="line", color=styleOpt["gaussianOutlierDistributionColor"]
            )
            gaussianOutlierDistributionSymbol.set_label("outlier dist.")
            legendSymbols.append(gaussianOutlierDistributionSymbol)

        ax.legend(handles=legendSymbols)

        # axis legend
        ax.set_xlabel(r"translational error in $m$")
        ax.set_ylabel(r"rotational error in $^\circ$")

    return fig, ax


def graspingErrorsHistogram(
    errors,
    correspondingMethods,
    correspondingModelNames,
    n_bins=20,
    mode="translational",
    plotLegend=True,
):
    # Ensure that the two lists have the same length
    assert len(errors) == len(correspondingMethods)

    # Create a defaultdict with lists as default values
    grouped_vals = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, val in zip(correspondingMethods, errors):
        grouped_vals[method].append(val)

    # Convert defaultdict to a regular dict (optional)
    grouped_vals = dict(grouped_vals)
    cols = []
    histogramColors = []
    for key in grouped_vals:
        cols.append(grouped_vals[key])
        histogramColors.append(styleOpt["methodColors"][key])
    x = np.vstack((cols)).T

    # x = x - np.mean(x, axis=0)
    # fig, ax = setupLatexPlot2D(
    #     figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    # )
    fig = plt.figure(figsize=(styleOpt["figureWidth"], styleOpt["figureHeight"]))
    ax = fig.add_subplot()
    hist_handle = ax.hist(
        x, n_bins, density=False, histtype="bar", color=histogramColors
    )

    # # plot gaussian
    x_axis = np.linspace(hist_handle[1][0], hist_handle[1][-1], 1000)
    fitLineStyle = styleOpt["histogramFitLineStyle"]
    scalefactor = len(x) * np.mean(np.diff(hist_handle[1]))
    for method in grouped_vals:
        # mu = np.mean(grouped_vals[method])
        # std = np.std(grouped_vals[method])
        # pdf = norm.pdf(x_axis, mu, std)
        # if mode == "translational":
        #     gamma_params = gamma.fit(grouped_vals[method], floc=0, fscale=0.01)
        # elif mode == "rotational":
        #     gamma_params = gamma.fit(grouped_vals[method], floc=0, fscale=0.01)
        # else:
        default_params = np.mean(
            np.stack(
                (
                    np.array(gamma.fit(grouped_vals["cpd"])),
                    np.array(gamma.fit(grouped_vals["spr"])),
                    np.array(gamma.fit(grouped_vals["kpr"])),
                )
            ),
            axis=0,
        )
        gamma_params = gamma.fit(grouped_vals[method], floc=0)
        print(
            "{} mean: {}".format(
                method,
                gamma.mean(
                    a=gamma_params[0], loc=gamma_params[1], scale=gamma_params[2]
                ),
            )
        )
        print(
            "{} std: {}".format(
                method,
                gamma.std(
                    a=gamma_params[0], loc=gamma_params[1], scale=gamma_params[2]
                ),
            )
        )
        pdf = gamma.pdf(x_axis, gamma_params[0], gamma_params[1], gamma_params[2])
        probability = pdf * np.mean(np.diff(hist_handle[1]))
        estimatedCounts = probability * len(x)
        # scale density to counts
        ax.plot(
            x_axis,
            estimatedCounts,
            color=styleOpt["methodColors"][method],
            linestyle=fitLineStyle,
        )

    if styleOpt["addSecondaryAxis"]:
        # add density as secondary axis
        def counts_to_probability(counts):
            return counts / len(x)

        def probability_to_counts(probability):
            return probability * len(x)

        secax_y = ax.secondary_yaxis(
            "right", functions=(counts_to_probability, probability_to_counts)
        )
        secax_y.set_ylabel(r"probability")

    # set axis labels
    if mode == "translational":
        ax.set_xlabel(r"translational errors in $m$")
    elif mode == "rotational":
        ax.set_xlabel(r"rotational error in $^\circ$")
    ax.set_ylabel("counts")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # make y axis only integers
    if plotLegend:
        # create legend
        legendSymbols = []
        for method in grouped_vals:
            legendSymbol = Patch(
                facecolor=styleOpt["methodColors"][method],
                label=method.upper(),
            )
            legendSymbols.append(legendSymbol)
        for method in grouped_vals:
            legendSymbol = Line2D(
                [],
                [],
                color=styleOpt["methodColors"][method],
                linestyle=fitLineStyle,
                label="fitted distribution",
            )
            legendSymbols.append(legendSymbol)
        ax.legend(handles=legendSymbols, ncol=2)
    return fig, ax


def outlierRatioBarPlot(
    translationalGraspingErrors,
    rotationalGraspingErrors,
    correspondingMethods,
    translationalThreshold=None,
    rotationalThreshold=None,
    barWidth=None,
    spacingFactor=None,
    methodsToEvaluate=None,
    mode="gaussian",  # "theshold", "gaussian"
):
    barWidth = 0.5 if barWidth is None else barWidth
    spacingFactor = 1 if spacingFactor is None else spacingFactor
    methodsToEvaluate = (
        controlOpt["methodsToEvaluate"]
        if methodsToEvaluate is None
        else methodsToEvaluate
    )
    # fig, ax = setupLatexPlot2D(
    #     figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    # )
    fig = plt.figure(figsize=(styleOpt["figureWidth"], styleOpt["figureHeight"]))
    ax = fig.add_subplot()

    inlierCount = {}
    outlierCount = {}
    if mode == "threshold":
        for method in methodsToEvaluate:
            inlierCount[method] = 0
            outlierCount[method] = 0
        for i, (transplationalError, rotationalError, method) in enumerate(
            zip(
                translationalGraspingErrors,
                rotationalGraspingErrors,
                correspondingMethods,
            )
        ):
            if (
                transplationalError > translationalThreshold
                or rotationalError > rotationalThreshold
            ):
                outlierCount[method] += 1
            else:
                inlierCount[method] += 1
    elif mode == "gaussian":
        # fit GMM
        data = np.stack(
            (np.array(translationalGraspingErrors), np.array(rotationalGraspingErrors)),
            axis=1,
        )
        gm = GaussianMixture(n_components=2, covariance_type="full")
        gm.fit(data)
        # Separate inlier and outlier Gaussians based on mean and variance (covariance)
        means = gm.means_
        covariances = gm.covariances_
        # Calculate the determinant of the covariance to find the one with the larger spread (outlier)
        determinants = np.array([np.linalg.det(cov) for cov in covariances])
        # Find the indices of the inlier and outlier Gaussians
        inlier_idx = np.argmin(determinants)  # Inlier has lower variance
        outlier_idx = np.argmax(determinants)  # Outlier has higher variance

        # determine correspondance for each data point
        correspondance = gm.predict(data)

        # count inliers and outliers for each method
        for method in methodsToEvaluate:
            inlierCount[method] = 0
            outlierCount[method] = 0
        for i, (transplationalError, rotationalError, method) in enumerate(
            zip(
                translationalGraspingErrors,
                rotationalGraspingErrors,
                correspondingMethods,
            )
        ):
            if correspondance[i] == outlier_idx:
                outlierCount[method] += 1
            else:
                inlierCount[method] += 1
    XTicks = np.arange(len(methodsToEvaluate)) * spacingFactor
    barColors = []
    for x, method in zip(XTicks, methodsToEvaluate):
        barValue = inlierCount[method]
        absValue = inlierCount[method] + outlierCount[method]
        bottomValue = barValue / absValue
        successRate = bottomValue * 100
        successBar = plt.bar(
            x, bottomValue, barWidth, color=styleOpt["methodColors"][method]
        )
        unsuccessBar = plt.bar(
            x,
            ((absValue - barValue) / absValue),
            barWidth,
            bottom=bottomValue,
            color="gray",
        )
        barColors.append(styleOpt["methodColors"][method])
        if styleOpt["plotSuccessRateText"]:
            for unsuccessRect, successRect in zip(unsuccessBar, successBar):
                height = unsuccessRect.get_height() + successRect.get_height()
                plt.text(
                    successRect.get_x() + successRect.get_width() / 2.0,
                    height,
                    "{:.1f}".format(successRate) + "\%",
                    ha="center",
                    va="bottom",
                )
    ax.set_xlabel("methods")
    ax.set_ylabel(r"success rate in \%")
    ax.set_xticks(XTicks, [x.upper() for x in methodsToEvaluate])
    ax.set_ylim([0, 1.3])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [r"$0$", r"$25$", r"$50$", r"75", r"100"])
    if styleOpt["plotSuccessRateLegend"]:
        ax.legend()
        pa1 = Patch(facecolor=barColors[0], edgecolor="black")
        pa2 = Patch(facecolor=barColors[1], edgecolor="black")
        pa3 = Patch(facecolor=barColors[2], edgecolor="black")
        pb1 = Patch(facecolor="gray", edgecolor="gray")
        ax.legend(
            handles=[pa1, pa2, pa3, pb1],
            labels=[
                methodsToEvaluate[0].upper(),
                methodsToEvaluate[1].upper(),
                methodsToEvaluate[2].upper(),
                "outlier",
            ],
            ncol=2,
            # handletextpad=0.5,
            # handlelength=1.0,
            # columnspacing=-0.5,
        )
    return fig, ax


def plotGraspingAccuracyBoxPlot(
    graspingErrors, correspondingMethods, medianColor=[0, 0, 0], mode="translational"
):
    # Ensure that the two lists have the same length
    assert len(graspingErrors) == len(correspondingMethods)

    # Create a defaultdict with lists as default values
    grouped_vals = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, val in zip(correspondingMethods, graspingErrors):
        grouped_vals[method].append(val)

    # Convert defaultdict to a regular dict (optional)
    grouped_vals = dict(grouped_vals)

    # fig, ax = setupLatexPlot2D(
    #     figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    # )
    fig = plt.figure(figsize=(styleOpt["figureWidth"], styleOpt["figureHeight"]))
    ax = fig.add_subplot()

    data = []
    labels = []
    colors = []

    facealpha = styleOpt["boxPlotFaceAlpha"]
    linealpha = 1

    for method in grouped_vals:
        data.append(grouped_vals[method])
        labels.append(method.upper())
        colors.append(styleOpt["methodColors"][method])

    boxPlot = ax.boxplot(
        data,
        patch_artist=True,  # fill with color
        labels=labels,
    )
    for median, color in zip(boxPlot["medians"], colors):
        median.set_color(medianColor)
    # Set the box colors
    for patch, color in zip(boxPlot["boxes"], colors):
        patch.set_facecolor(np.concatenate((color, [facealpha])))
        patch.set_edgecolor(np.concatenate((color, [linealpha])))
    # Set the flier colors
    for flier, color in zip(boxPlot["fliers"], colors):
        flier.set(
            marker="o", color=color, markersize=6, markerfacecolor=color + (facealpha,)
        )
    if mode == "translational":
        ax.set_ylabel(r"translational error in $m$")
    elif mode == "rotational":
        ax.set_ylabel(r"rotational error in $\circ$")
    else:
        raise NotImplementedError
    return fig, ax


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        resultsToEvaluate = resultFolderPaths
    else:
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]

    graspingAccuracyResults = []
    translationalGraspingErrors = []
    rotationalGraspingErrors = []
    methods = []
    dataSets = []
    models = []
    grasps = []
    plotColors = []
    plotMarkers = []
    for nResult, resultFolderPath in enumerate(resultsToEvaluate):
        resultFilePath = os.path.join(resultFolderPath, resultFileName)
        result = eval.loadResults(resultFilePath)

        existingMethods = eval.getRegistrationMethods(result)
        methodsToEvaluate = [
            method
            for method in existingMethods
            if method in controlOpt["methodsToEvaluate"]
        ]
        for nMethod, method in enumerate(methodsToEvaluate):
            numRegistrationResults = eval.getNumRegistrationResults(result)
            if controlOpt["registrationResultsToEvaluate"][0] == -1:
                registrationResultsToEvaluate = list(
                    range(
                        0, numRegistrationResults - 1
                    )  # do not evaluate last registration result since this is only the final frame
                )
            else:
                registrationResultsToEvaluate = controlOpt[
                    "registrationResultsToEvaluate"
                ]
            for nRegistrationResult in registrationResultsToEvaluate:
                graspingAccuracyError = evaluateGraspingAccuracy(
                    result, method, nRegistrationResult
                )
                graspingAccuracyResults.append(graspingAccuracyError)
                translationalGraspingErrors.append(
                    graspingAccuracyError["graspingPositionErrors"]
                )
                rotationalGraspingErrors.append(
                    graspingAccuracyError["graspingAngularErrorsInGrad"]
                )
                methods.append(method)
                grasps.append(nRegistrationResult)
                modelName = result["dataSetName"].split("_")[-1]
                models.append(modelName)
                plotColors.append(styleOpt["methodColors"][method])
                plotMarkers.append(styleOpt["modelMarkers"][modelName])
    # if controlOpt["showPlot"]:

    # if controlOpt["save"]:
    #     dataSetName = result["dataSetName"]
    #     fileID = "_".join(
    #         result["trackingResults"][method]["registrationResults"][
    #             nRegistrationResult
    #         ]["fileName"].split("_")[0:3]
    #     )
    #     fileName = fileID + "_" + saveOpt["saveFileName"]
    #     saveFolderPath = saveOpt["saveFolder"]
    #     saveFolderPath = os.path.join(saveFolderPath, dataSetName, method)
    #     saveFilePath = os.path.join(saveFolderPath, fileName)
    #     if not os.path.exists(saveFolderPath):
    #         os.makedirs(saveFolderPath, exist_ok=True)
    #     eval.saveImage(rgbImg, saveFilePath)
    #     if controlOpt["verbose"]:
    #         print(
    #             "Saved registration {}/{} from method {}/{} of result {}/{} at {}".format(
    #                 nRegistrationResult + 1,
    #                 len(registrationResultsToEvaluate),
    #                 nMethod + 1,
    #                 len(methodsToEvaluate),
    #                 nResult + 1,
    #                 len(resultsToEvaluate),
    #                 saveFilePath,
    #             )
    #         )
    meanTranslationalErrors = np.mean(translationalGraspingErrors)
    stdTranslationalErrors = np.std(translationalGraspingErrors)
    meanRotationalErrors = np.mean(rotationalGraspingErrors)
    stdRotationalErrors = np.std(rotationalGraspingErrors)

    if controlOpt["makeScatterPlot"]:
        fig_scatterPlot, ax_scatterPlot = scatterPlotGraspingErrors(
            translationalGraspingErrors=translationalGraspingErrors,
            rotationalGraspingErrors=rotationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            colors=plotColors,
            markers=plotMarkers,
            alpha=styleOpt["alpha"],
            translationalThreshold=styleOpt["translationalErrorThreshold"],
            rotationalThreshold=styleOpt["rotationalErrorThreshold"],
            translationalThresholdLineColor=styleOpt["translationalThresholdLineColor"],
            rotationalThresholdLineColor=styleOpt["rotationalThresholdLineColor"],
            thresholdLineStyle=styleOpt["thresholdLineStyle"],
        )
        if controlOpt["save"]:
            saveFileNameScatter = saveOpt["saveFileNameScatterPlot"]
            saveFolder = saveOpt["saveFolder"]
            savePathScatter = os.path.join(saveFolder, saveFileNameScatter)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            if controlOpt["saveAs"] == "png":
                fig_scatterPlot.savefig(savePathScatter)
            # save as tixfigure
            if controlOpt["saveAs"] == "TIKZ":
                tikzplotlib_fix_ncols(fig_scatterPlot)
                tikzplotlib.save(
                    figure=fig_scatterPlot,
                    filepath=savePathScatter + ".tex",
                )
            if controlOpt["saveAs"] == "PGF":
                fig_scatterPlot.savefig(
                    savePathScatter + ".pgf",
                    bbox_inches="tight",
                )
            if controlOpt["saveAs"] == "PDF":
                fig_scatterPlot.savefig(savePathScatter + ".pdf", pad_inches=0.0)

        # if controlOpt["showPlot"]:
        #     plt.show(block=True)
    if controlOpt["makeOutlierRatioBarPlot"]:
        fig_success_rate_plot, _ = outlierRatioBarPlot(
            translationalGraspingErrors=translationalGraspingErrors,
            rotationalGraspingErrors=rotationalGraspingErrors,
            correspondingMethods=methods,
            translationalThreshold=styleOpt["translationalErrorThreshold"],
            rotationalThreshold=styleOpt["rotationalErrorThreshold"],
        )
        if controlOpt["save"]:
            saveFileNameOutlierRatio = saveOpt["saveFileNameOutlierRatio"]
            saveFolder = saveOpt["saveFolder"]
            savePathOutlierRatio = os.path.join(saveFolder, saveFileNameOutlierRatio)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            if controlOpt["saveAs"] == "png":
                fig_success_rate_plot.savefig(savePathOutlierRatio, bbox_inches="tight")
            # save as tixfigure
            if controlOpt["saveAs"] == "TIKZ":
                tikzplotlib_fix_ncols(fig_success_rate_plot)
                tikzplotlib.save(
                    figure=fig_success_rate_plot,
                    filepath=savePathScatter + ".tex",
                )
            if controlOpt["saveAs"] == "PGF":
                fig_success_rate_plot.savefig(
                    savePathOutlierRatio + ".pgf",
                    bbox_inches="tight",
                )
            if controlOpt["saveAs"] == "PDF":
                fig_success_rate_plot.savefig(
                    savePathOutlierRatio + ".pdf", pad_inches=0.0
                )
    # histograms
    if controlOpt["makeHistogramPlot"]:
        fig_histogram_trans, ax_histogram_trans = graspingErrorsHistogram(
            errors=translationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            mode="translational",
        )
        fig_histogram_rot, ax_histogram_rot = graspingErrorsHistogram(
            errors=rotationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            mode="rotational",
            plotLegend=False,
        )
        if controlOpt["save"]:
            saveFileNameTrans = saveOpt["saveFileNameHistogramTranslational"]
            saveFileNameRot = saveOpt["saveFileNameHistogramRotaional"]
            saveFolder = saveOpt["saveFolder"]
            savePathTrans = os.path.join(saveFolder, saveFileNameTrans)
            savePathRot = os.path.join(saveFolder, saveFileNameRot)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            if controlOpt["saveAs"] == "png":
                fig_histogram_trans.savefig(savePathTrans)
                fig_histogram_rot.savefig(savePathRot)
            if controlOpt["saveAs"] == "PGF":
                fig_histogram_trans.savefig(savePathTrans + ".pgf", bbox_inches="tight")
                fig_histogram_rot.savefig(savePathRot + ".pgf", bbox_inches="tight")
            # save as tixfigure
            if controlOpt["saveAs"] == "TIKZ":
                tikzplotlib_fix_ncols(fig_histogram_trans)
                tikzplotlib.save(
                    figure=fig_histogram_trans, filepath=savePathTrans + ".tex"
                )
                tikzplotlib_fix_ncols(fig_histogram_rot)
                tikzplotlib.save(
                    figure=fig_histogram_rot, filepath=savePathRot + ".tex"
                )
            if controlOpt["saveAs"] == "PDF":
                fig_histogram_trans.savefig(savePathTrans + ".pdf", pad_inches=0.0)
                fig_histogram_rot.savefig(savePathRot + ".pdf", pad_inches=0.0)
    # box plots
    if controlOpt["makeGraspingAccuracyBoxPlot"]:
        fig_box_trans, _ = plotGraspingAccuracyBoxPlot(
            graspingErrors=translationalGraspingErrors,
            correspondingMethods=methods,
            mode="translational",
        )
        fig_box_rot, _ = plotGraspingAccuracyBoxPlot(
            graspingErrors=rotationalGraspingErrors,
            correspondingMethods=methods,
            mode="rotational",
        )
        if controlOpt["save"]:
            saveFileNameTrans = saveOpt["saveFileNameBoxPlotTranslational"]
            saveFileNameRot = saveOpt["saveFileNameBoxPlotRotational"]
            saveFolder = saveOpt["saveFolder"]
            savePathTrans = os.path.join(saveFolder, saveFileNameTrans)
            savePathRot = os.path.join(saveFolder, saveFileNameRot)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            if controlOpt["saveAs"] == "png":
                fig_box_trans.savefig(savePathTrans)
                fig_box_rot.savefig(savePathRot)
            if controlOpt["saveAs"] == "PGF":
                fig_box_trans.savefig(savePathTrans + ".pgf", bbox_inches="tight")
                fig_box_rot.savefig(savePathRot + ".pgf", bbox_inches="tight")
            if controlOpt["saveAs"] == "PDF":
                fig_box_trans.savefig(savePathTrans + ".pdf", pad_inches=0.0)
                fig_box_rot.savefig(savePathRot + ".pdf", pad_inches=0.0)
    if controlOpt["showPlot"]:
        plt.show(block=True)
    if controlOpt["verbose"]:
        print("Finished result generation.")
