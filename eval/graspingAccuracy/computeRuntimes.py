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
    "resultsToLoad": [-1],
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],
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
        runtimes = {}
        for nMethod, method in enumerate(methodsToEvaluate):
            runtimes[method] = {}
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
                runtimes[method]["grasp_" + str(nRegistrationResult)] = {}
                runtimes[method]["grasp_" + str(nRegistrationResult)]["data"] = result[
                    "trackingResults"
                ][method]["registrationResults"][2]["result"]["runtimes"][
                    "runtimesPerIteration"
                ]
                runtimes[method]["grasp_" + str(nRegistrationResult)]["total"] = np.sum(
                    runtimes[method]["grasp_" + str(nRegistrationResult)]["data"]
                )

            average_runtime_per_grasp = 0
            for nRegistrationResult in registrationResultsToEvaluate:
                average_runtime_per_grasp += runtimes[method][
                    "grasp_" + str(nRegistrationResult)
                ]["total"]
            runtimes[method]["average_runtime_per_grasp"] = (
                1 / len(registrationResultsToEvaluate) * average_runtime_per_grasp
            )
    for method in methodsToEvaluate:
        print(
            "Average runtime for method {} is: {}".format(
                method, runtimes[method]["average_runtime_per_grasp"]
            )
        )
    print("Done.")
