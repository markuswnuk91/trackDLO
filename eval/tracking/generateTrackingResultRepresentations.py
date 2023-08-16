import sys
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import (
        TrackingEvaluation,
    )
except:
    print("Imports for Tracking Result Evaluation failed.")
    raise


# script control parameters
controlOptions = {}
resultRootFolderPath = "data/eval/graspingAccuracy/results"
resultFileName = "result"
resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY"
]
resultsToLoad = [0]


def tabularizeResults(results, methodsToPrint=["cpd", "spr", "krcpd"]):
    trackingErrorScaleFactor = 100  # cm
    geometricErrorScaleFactor = 100  # cm
    runtimeScaleFactor = 1000  # mm
    for result in results:
        modelID = eval.getModelID(
            result["initializationResult"]["modelParameters"]["modelInfo"]["name"]
        )
        dataSetTableRowEntries = {
            "modelID": modelID,
            "n_tracked_frames": len(result["trackingResults"][0]["frames"]),
            "n_labeld_frames": len(
                result["trackingEvaluationResults"]["cpd"]["reprojectionErrors"][
                    "frames"
                ]
            ),
            "n_methods": len(methodsToPrint),
        }
        for trackingMethod in list(result["trackingEvaluationResults"].keys()):
            if trackingMethod not in methodsToPrint:
                continue
            dataSetTableRowEntries[trackingMethod] = {
                "trackingError": trackingErrorScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "trackingErrors"
                    ]
                ),
                "geometricError": geometricErrorScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "geometricErrors"
                    ]["accumulated"]
                ),
                "reprojectionError": np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "reprojectionErrors"
                    ]["mean"]
                ),
                "runtime": runtimeScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "runtimeResults"
                    ]["runtimesPerIteration"]
                ),
            }

        dataSetTableRow = f"""----------Cut-------------
            \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["modelID"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_tracked_frames"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_labeld_frames"]}$}} & \\acs{{CPD}} & ${dataSetTableRowEntries["cpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["cpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["cpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["cpd"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{SPR}} & ${dataSetTableRowEntries["spr"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["spr"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["spr"]["reprojectionError"])}$ & ${dataSetTableRowEntries["spr"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{KPR}} & ${dataSetTableRowEntries["krcpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["krcpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["krcpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["krcpd"]["runtime"]:.2f}$ \\\\
            ----------Cut-------------"""
        print(dataSetTableRow)
    return 0


if __name__ == "__main__":
    # setup eval class
    if resultsToLoad[0] == -1:
        resultsToEvaluate = resultFolderPaths
    else:
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if i in resultsToLoad
        ]

    results = []
    for resultFolderPath in resultsToEvaluate:
        # set file paths
        dataSetName = resultFolderPath.split("/")[-1]
        resultFilePath = resultFolderPath + "/" + resultFileName + ".pkl"

        # select config
        configFilePath = "/evalConfigs/evalConfig" + "_" + dataSetName + ".json"

        # setup evalulation
        global eval
        pathToConfigFile = os.path.dirname(os.path.abspath(__file__)) + configFilePath
        eval = TrackingEvaluation(configFilePath=pathToConfigFile)

        # load result for the data set
        dataSetResult = eval.loadResults(resultFilePath)
        results.append(dataSetResult)

    # create latex table with results
    tabularizeResults(results)
