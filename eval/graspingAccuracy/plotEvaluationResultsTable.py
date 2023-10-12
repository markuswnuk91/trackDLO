import sys
import os
import numpy as np
from collections import defaultdict

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
    "resultsToLoad": [-1],
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],
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


def printTable(
    translationalErrors,
    rotationalErrors,
    correspondingMethods,
    translationalScaleFactor=100,
    rotationalScaleFactor=1,
):
    # Ensure that the two lists have the same length
    assert len(translationalErrors) == len(correspondingMethods)
    assert len(rotationalErrors) == len(correspondingMethods)

    evaluationResults = {
        "nGrasps": len(correspondingMethods) / len(set(correspondingMethods)),
        "translationalResults": {},
        "rotationalResults": {},
    }
    evaluationResults_trans = defaultdict(list)
    evaluationResults_rot = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, translationalError, rotationalError in zip(
        correspondingMethods, translationalErrors, rotationalErrors
    ):
        evaluationResults_trans[method].append(translationalError)
        evaluationResults_rot[method].append(rotationalError)

    for method in evaluationResults_trans:
        evaluationResults["translationalResults"][method] = {
            "mean": np.mean(evaluationResults_trans[method]),
            "std": np.std(evaluationResults_trans[method]),
        }
        evaluationResults["rotationalResults"][method] = {
            "mean": np.mean(evaluationResults_rot[method]),
            "std": np.std(evaluationResults_rot[method]),
        }

    # generate table
    formattedResults = {
        "nGrasps": str(int(evaluationResults["nGrasps"])),
        "trans": {},
        "rot": {},
    }
    methodOfMinTranslationalMean = min(
        evaluationResults["translationalResults"],
        key=lambda x: evaluationResults["translationalResults"][x]["mean"],
    )
    methodOfMinTranslationalStd = min(
        evaluationResults["translationalResults"],
        key=lambda x: evaluationResults["translationalResults"][x]["std"],
    )
    methodOfMinRotationalMean = min(
        evaluationResults["rotationalResults"],
        key=lambda x: evaluationResults["rotationalResults"][x]["mean"],
    )
    methodOfMinRotationalStd = min(
        evaluationResults["rotationalResults"],
        key=lambda x: evaluationResults["rotationalResults"][x]["std"],
    )
    for method in list(evaluationResults["translationalResults"].keys()):
        formattedResults["trans"][method] = {}
        formattedResults["rot"][method] = {}
        if method == methodOfMinTranslationalMean:
            formattedResults["trans"][method][
                "mean"
            ] = f"""\\textbf{{{format(
                evaluationResults["translationalResults"][method]["mean"]
                * translationalScaleFactor,
                ".1f")}}}"""
        else:
            formattedResults["trans"][method]["mean"] = format(
                evaluationResults["translationalResults"][method]["mean"]
                * translationalScaleFactor,
                ".1f",
            )
        if method == methodOfMinTranslationalStd:
            formattedResults["trans"][method][
                "std"
            ] = f"""\\textbf{{{format(
                evaluationResults["translationalResults"][method]["std"]
                * translationalScaleFactor,
                ".1f")}}}"""
        else:
            formattedResults["trans"][method]["std"] = format(
                evaluationResults["translationalResults"][method]["std"]
                * translationalScaleFactor,
                ".1f",
            )
        if method == methodOfMinRotationalMean:
            formattedResults["rot"][method][
                "mean"
            ] = f"""\\textbf{{{format(
                evaluationResults["rotationalResults"][method]["mean"]
                * rotationalScaleFactor,
                ".1f")}}}"""
        else:
            formattedResults["rot"][method]["mean"] = format(
                evaluationResults["rotationalResults"][method]["mean"]
                * rotationalScaleFactor,
                ".1f",
            )
        if method == methodOfMinRotationalStd:
            formattedResults["rot"][method][
                "std"
            ] = f"""\\textbf{{{format(
                evaluationResults["rotationalResults"][method]["std"]
                * rotationalScaleFactor,
                ".1f")}}}"""
        else:
            formattedResults["rot"][method]["std"] = format(
                evaluationResults["rotationalResults"][method]["std"]
                * rotationalScaleFactor,
                ".1f",
            )

    table_head = r"""
        & & \multicolumn{2}{c}{\acs{CPD}} & \multicolumn{2}{c}{\acs{SPR}} & \multicolumn{2}{c}{\acs{KPR}}
        \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
        error           & $n_{\text{grasp}}$  & $\mu$ & $\sigma$ & $\mu$ & $\sigma$ & $\mu$ & $\sigma$ \\\midrule"""
    table_body = f"""
        $\\nicefrac{{e_\\text{{trans}}}}{{\\si{{cm}}}}$ & \multirow{{2}}{{*}}{{{formattedResults["nGrasps"]}}} & {formattedResults["trans"]["cpd"]["mean"]} & {formattedResults["trans"]["cpd"]["std"]} & {formattedResults["trans"]["spr"]["mean"]} & {formattedResults["trans"]["spr"]["std"]} & {formattedResults["trans"]["kpr"]["mean"]} & {formattedResults["trans"]["kpr"]["std"]}\\\\
        $\\nicefrac{{e_\\text{{rot}}}}{{\\si{{rad}}}}$ &   & {formattedResults["rot"]["cpd"]["mean"]} & {formattedResults["rot"]["cpd"]["std"]} & {formattedResults["rot"]["spr"]["mean"]} & {formattedResults["rot"]["spr"]["std"]} & {formattedResults["rot"]["kpr"]["mean"]} & {formattedResults["rot"]["kpr"]["std"]}\\\\
        """
    table_end = r"""\bottomrule"""
    table = table_head + table_body + table_end
    print(table)
    return table


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
                    graspingAccuracyError["graspingAngularErrorsInRad"]
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
    printTable(
        translationalErrors=translationalGraspingErrors,
        rotationalErrors=rotationalGraspingErrors,
        correspondingMethods=methods,
    )
