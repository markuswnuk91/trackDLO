import sys
import os
import numpy as np
from collections import defaultdict
from scipy.stats import norm, gamma
from sklearn.mixture import GaussianMixture

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
except:
    print("Imports for plotting script grasping accuracy evaluation table failed.")
    raise

global eval
eval = GraspingAccuracyEvaluation()

controlOpt = {
    "resultsToLoad": [-1],  # -1: all
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],  # -1: all
    "scale_translational_errors": 100,  # convert translational errors to cm
}

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


def fitGammaAndCalculateMeanAndStd(errors):
    params = gamma.fit(errors, floc=0)
    mean = gamma.mean(a=params[0], loc=params[1], scale=params[2])
    std = gamma.std(a=params[0], loc=params[1], scale=params[2])
    result = {}
    result["parameters"] = params
    result["mean"] = mean
    result["std"] = std
    return result


def calculateSuccessRate(
    translationalGraspingErrors,
    rotationalGraspingErrors,
    correspondingMethods,
    methodToEvaluate,
):
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
    inlierCount = {}
    outlierCount = {}
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

    successRate = inlierCount[methodToEvaluate] / (
        inlierCount[methodToEvaluate] + outlierCount[methodToEvaluate]
    )
    successRate *= 100  # convert in percent
    return successRate


def printTable(statisticalEvaluationResults):

    table_header = r"""& & \multicolumn{2}{c}{translational errors} &\multicolumn{2}{c}{rotational errors} & \\ \cmidrule(lr){3-4} \cmidrule(lr){5-6}
	method & $n_{\text{grasps}}$ & 
	$ \left(\tilde{e}_{\text{trans}}\right)$ & $\bar{e}_{\text{trans}}^{\left(\Gamma\right)}$ & 
	$ \left(\tilde{e}_{\text{rot}}\right)$ & $\bar{e}_{\text{rot}}^{\left(\Gamma\right)}$ & 
	success rate \\
	\midrule
    -------------------------------------------------------------------
    """
    table_body = """\n"""
    for i, method in enumerate(controlOpt["methodsToEvaluate"]):
        statisticalEvaluationResults[method]
        method_string = method.upper()
        n_grasps = len(statisticalEvaluationResults[method]["translational"]["data"])
        if i == 0:
            grasp_str = (
                f"""\\multirow{{{len(methodsToEvaluate)}}}{{*}}{{{int(n_grasps)}}}"""
            )
        else:
            grasp_str = f""
        median_trans = (
            statisticalEvaluationResults[method]["translational"]["median"]
            * controlOpt["scale_translational_errors"]
        )
        iqr_trans_low = (
            statisticalEvaluationResults[method]["translational"]["median"]
            - statisticalEvaluationResults[method]["translational"]["q1"]
        ) * controlOpt["scale_translational_errors"]
        iqr_trans_up = (
            statisticalEvaluationResults[method]["translational"]["q3"]
            - statisticalEvaluationResults[method]["translational"]["median"]
        ) * controlOpt["scale_translational_errors"]

        mean_trans = (
            statisticalEvaluationResults[method]["translational"]["mean"]
            * controlOpt["scale_translational_errors"]
        )
        std_trans = (
            statisticalEvaluationResults[method]["translational"]["std"]
            * controlOpt["scale_translational_errors"]
        )
        median_rot = statisticalEvaluationResults[method]["rotational"]["median"]
        # iqr_rot = statisticalEvaluationResults[method]["rotational"]["iqr"]
        iqr_rot_low = (
            statisticalEvaluationResults[method]["rotational"]["median"]
            - statisticalEvaluationResults[method]["rotational"]["q1"]
        )
        iqr_rot_up = (
            statisticalEvaluationResults[method]["rotational"]["q3"]
            - statisticalEvaluationResults[method]["rotational"]["median"]
        )
        mean_rot = statisticalEvaluationResults[method]["rotational"]["mean"]
        std_rot = statisticalEvaluationResults[method]["rotational"]["std"]
        successRate = statisticalEvaluationResults[method]["successRate"]
        table_body += f"""\\ac{{{method_string}}} & {grasp_str} & ${median_trans:.1f}(-{iqr_trans_low:.1f},+{iqr_trans_up:.1f}) $ & ${mean_trans:.1f} \\pm {std_trans:.1f} $ & ${median_rot:.1f}(-{iqr_rot_low:.1f},+{iqr_rot_up:.1f}) $ & ${mean_rot:.1f} \\pm {std_rot:.1f}$ & ${successRate:.1f}$ \\\\ \n"""
    # \ac{CPD} & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx \\
    # \ac{SPR} & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx \\
    # \ac{KPR} & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx & xx.xx

    table_end = r"""\bottomrule"""
    print(table_header + table_body + table_end)
    return


def printRelativeImprvementsOverSPR(statisticalEvaluationResults):
    print("Relative improvements for translational errors:")

    # MEAN
    # relative_improvement_mean = (
    #     statisticalEvaluationResults["spr"]["translational"]["mean"]
    #     - statisticalEvaluationResults["kpr"]["translational"]["mean"]
    # ) / statisticalEvaluationResults["spr"]["translational"]["mean"]

    # STD
    # relative_improvement_std = (
    #     statisticalEvaluationResults["spr"]["translational"]["std"]
    #     - statisticalEvaluationResults["kpr"]["translational"]["std"]
    # ) / statisticalEvaluationResults["spr"]["translational"]["std"]

    # Median
    relative_improvement_median_trans = (
        statisticalEvaluationResults["spr"]["translational"]["median"]
        - statisticalEvaluationResults["kpr"]["translational"]["median"]
    ) / statisticalEvaluationResults["spr"]["translational"]["median"]
    print(
        "Translational improvements Median: {}".format(
            relative_improvement_median_trans
        )
    )

    # IQR
    relative_improvement_iqr_trans = (
        statisticalEvaluationResults["spr"]["translational"]["iqr"]
        - statisticalEvaluationResults["kpr"]["translational"]["iqr"]
    ) / statisticalEvaluationResults["spr"]["translational"]["iqr"]
    print("Translational improvements IQR: {}".format(relative_improvement_iqr_trans))

    print("Relative improvements for rotational errors:")
    # MEAN
    # relative_improvement_mean = (
    #     statisticalEvaluationResults["spr"]["rotational"]["mean"]
    #     - statisticalEvaluationResults["kpr"]["rotational"]["mean"]
    # ) / statisticalEvaluationResults["spr"]["rotational"]["mean"]

    # STD
    # relative_improvement_std = (
    #     statisticalEvaluationResults["spr"]["rotational"]["std"]
    #     - statisticalEvaluationResults["kpr"]["rotational"]["std"]
    # ) / statisticalEvaluationResults["spr"]["rotational"]["std"]

    # Median
    relative_improvement_median_rot = (
        statisticalEvaluationResults["spr"]["rotational"]["median"]
        - statisticalEvaluationResults["kpr"]["rotational"]["median"]
    ) / statisticalEvaluationResults["spr"]["rotational"]["median"]
    print("Rotational improvements Median: {}".format(relative_improvement_median_rot))

    # IQR
    relative_improvement_iqr_rot = (
        statisticalEvaluationResults["spr"]["rotational"]["iqr"]
        - statisticalEvaluationResults["kpr"]["rotational"]["iqr"]
    ) / statisticalEvaluationResults["spr"]["rotational"]["iqr"]
    print("Rotational improvements IQR: {}".format(relative_improvement_iqr_rot))


# def printTable(
#     translationalErrors,
#     rotationalErrors,
#     correspondingMethods,
#     translationalScaleFactor=100,
#     rotationalScaleFactor=1,
# ):
#     # Ensure that the two lists have the same length
#     assert len(translationalErrors) == len(correspondingMethods)
#     assert len(rotationalErrors) == len(correspondingMethods)

#     evaluationResults = {
#         "nGrasps": len(correspondingMethods) / len(set(correspondingMethods)),
#         "translationalResults": {},
#         "rotationalResults": {},
#     }
#     evaluationResults_trans = defaultdict(list)
#     evaluationResults_rot = defaultdict(list)

#     # Iterate through both lists simultaneously using zip
#     for method, translationalError, rotationalError in zip(
#         correspondingMethods, translationalErrors, rotationalErrors
#     ):
#         evaluationResults_trans[method].append(translationalError)
#         evaluationResults_rot[method].append(rotationalError)

#     for method in evaluationResults_trans:
#         evaluationResults["translationalResults"][method] = {
#             "mean": np.mean(evaluationResults_trans[method]),
#             "std": np.std(evaluationResults_trans[method]),
#         }
#         evaluationResults["rotationalResults"][method] = {
#             "mean": np.mean(evaluationResults_rot[method]),
#             "std": np.std(evaluationResults_rot[method]),
#         }

#     # generate table
#     formattedResults = {
#         "nGrasps": str(int(evaluationResults["nGrasps"])),
#         "trans": {},
#         "rot": {},
#     }
#     methodOfMinTranslationalMean = min(
#         evaluationResults["translationalResults"],
#         key=lambda x: evaluationResults["translationalResults"][x]["mean"],
#     )
#     methodOfMinTranslationalStd = min(
#         evaluationResults["translationalResults"],
#         key=lambda x: evaluationResults["translationalResults"][x]["std"],
#     )
#     methodOfMinRotationalMean = min(
#         evaluationResults["rotationalResults"],
#         key=lambda x: evaluationResults["rotationalResults"][x]["mean"],
#     )
#     methodOfMinRotationalStd = min(
#         evaluationResults["rotationalResults"],
#         key=lambda x: evaluationResults["rotationalResults"][x]["std"],
#     )
#     for method in list(evaluationResults["translationalResults"].keys()):
#         formattedResults["trans"][method] = {}
#         formattedResults["rot"][method] = {}
#         if method == methodOfMinTranslationalMean:
#             formattedResults["trans"][method][
#                 "mean"
#             ] = f"""\\textbf{{{format(
#                 evaluationResults["translationalResults"][method]["mean"]
#                 * translationalScaleFactor,
#                 ".1f")}}}"""
#         else:
#             formattedResults["trans"][method]["mean"] = format(
#                 evaluationResults["translationalResults"][method]["mean"]
#                 * translationalScaleFactor,
#                 ".1f",
#             )
#         if method == methodOfMinTranslationalStd:
#             formattedResults["trans"][method][
#                 "std"
#             ] = f"""\\textbf{{{format(
#                 evaluationResults["translationalResults"][method]["std"]
#                 * translationalScaleFactor,
#                 ".1f")}}}"""
#         else:
#             formattedResults["trans"][method]["std"] = format(
#                 evaluationResults["translationalResults"][method]["std"]
#                 * translationalScaleFactor,
#                 ".1f",
#             )
#         if method == methodOfMinRotationalMean:
#             formattedResults["rot"][method][
#                 "mean"
#             ] = f"""\\textbf{{{format(
#                 evaluationResults["rotationalResults"][method]["mean"]
#                 * rotationalScaleFactor,
#                 ".1f")}}}"""
#         else:
#             formattedResults["rot"][method]["mean"] = format(
#                 evaluationResults["rotationalResults"][method]["mean"]
#                 * rotationalScaleFactor,
#                 ".1f",
#             )
#         if method == methodOfMinRotationalStd:
#             formattedResults["rot"][method][
#                 "std"
#             ] = f"""\\textbf{{{format(
#                 evaluationResults["rotationalResults"][method]["std"]
#                 * rotationalScaleFactor,
#                 ".1f")}}}"""
#         else:
#             formattedResults["rot"][method]["std"] = format(
#                 evaluationResults["rotationalResults"][method]["std"]
#                 * rotationalScaleFactor,
#                 ".1f",
#             )

#     table_head = r"""
#         & & \multicolumn{2}{c}{\acs{CPD}} & \multicolumn{2}{c}{\acs{SPR}} & \multicolumn{2}{c}{\acs{KPR}}
#         \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
#         error           & $n_{\text{grasp}}$  & $\mu$ & $\sigma$ & $\mu$ & $\sigma$ & $\mu$ & $\sigma$ \\\midrule"""
#     table_body = f"""
#         $\\nicefrac{{e_\\text{{trans}}}}{{\\si{{cm}}}}$ & \multirow{{2}}{{*}}{{{formattedResults["nGrasps"]}}} & {formattedResults["trans"]["cpd"]["mean"]} & {formattedResults["trans"]["cpd"]["std"]} & {formattedResults["trans"]["spr"]["mean"]} & {formattedResults["trans"]["spr"]["std"]} & {formattedResults["trans"]["kpr"]["mean"]} & {formattedResults["trans"]["kpr"]["std"]}\\\\
#         $\\nicefrac{{e_\\text{{rot}}}}{{\\si{{rad}}}}$ &   & {formattedResults["rot"]["cpd"]["mean"]} & {formattedResults["rot"]["cpd"]["std"]} & {formattedResults["rot"]["spr"]["mean"]} & {formattedResults["rot"]["spr"]["std"]} & {formattedResults["rot"]["kpr"]["mean"]} & {formattedResults["rot"]["kpr"]["std"]}\\\\
#         """
#     table_end = r"""\bottomrule"""
#     table = table_head + table_body + table_end
#     print(table)
#     return table


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
                graspingAccuracyError = eval.evaluateGraspingAccuracy(
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

    # create result dict of structure
    # statitsticalEvaluationResults[method][statisticalMeasure]
    statisticalEvaluationResults = {}
    for method in methodsToEvaluate:
        statisticalEvaluationResults[method] = {}
        statisticalEvaluationResults[method]["translational"] = {}
        statisticalEvaluationResults[method]["rotational"] = {}
        data_translational = eval.filterAccumulatedGraspingResultsByMethod(
            np.array(translationalGraspingErrors), np.array(methods), method
        )
        data_rotational = eval.filterAccumulatedGraspingResultsByMethod(
            np.array(rotationalGraspingErrors), np.array(methods), method
        )
        statisticalEvaluationResults[method]["translational"][
            "data"
        ] = data_translational
        statisticalEvaluationResults[method]["rotational"]["data"] = data_rotational

        # calculate median and IRQ
        median_tranlsational = np.median(data_translational)
        median_rotational = np.median(data_rotational)

        statisticalEvaluationResults[method]["translational"][
            "median"
        ] = median_tranlsational
        statisticalEvaluationResults[method]["rotational"]["median"] = median_rotational
        q1_tranlsational = np.percentile(data_translational, 25)
        q3_tranlsational = np.percentile(data_translational, 75)
        iqr_translational = np.subtract(*np.percentile(data_translational, [75, 25]))
        statisticalEvaluationResults[method]["translational"]["iqr"] = iqr_translational
        statisticalEvaluationResults[method]["translational"]["q1"] = q1_tranlsational
        statisticalEvaluationResults[method]["translational"]["q3"] = q3_tranlsational

        q1_rotational = np.percentile(data_rotational, 25)
        q3_rotational = np.percentile(data_rotational, 75)
        iqr_rotational = np.subtract(*np.percentile(data_rotational, [75, 25]))
        statisticalEvaluationResults[method]["rotational"]["iqr"] = iqr_rotational
        statisticalEvaluationResults[method]["rotational"]["q1"] = q1_rotational
        statisticalEvaluationResults[method]["rotational"]["q3"] = q3_rotational

        # fit gamma disribution
        gamma_fit_result_translational = fitGammaAndCalculateMeanAndStd(
            data_translational
        )
        gamma_fit_result_rotational = fitGammaAndCalculateMeanAndStd(data_rotational)

        statisticalEvaluationResults[method]["translational"]["mean"] = (
            gamma_fit_result_translational["mean"]
        )
        statisticalEvaluationResults[method]["translational"]["std"] = (
            gamma_fit_result_translational["std"]
        )
        statisticalEvaluationResults[method]["rotational"]["mean"] = (
            gamma_fit_result_rotational["mean"]
        )
        statisticalEvaluationResults[method]["rotational"]["std"] = (
            gamma_fit_result_rotational["std"]
        )
        # calculate median
        # calculate mean and std (by fitting gamma distribution)
        successRate = calculateSuccessRate(
            translationalGraspingErrors=translationalGraspingErrors,
            rotationalGraspingErrors=rotationalGraspingErrors,
            correspondingMethods=methods,
            methodToEvaluate=method,
        )
        statisticalEvaluationResults[method]["successRate"] = successRate
    printTable(statisticalEvaluationResults=statisticalEvaluationResults)
    printRelativeImprvementsOverSPR(
        statisticalEvaluationResults=statisticalEvaluationResults
    )
    # printTable(
    #     translationalErrors=translationalGraspingErrors,
    #     rotationalErrors=rotationalGraspingErrors,
    #     correspondingMethods=methods,
    # )
    print("Done.")
