import sys
import os
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
except:
    print("Imports for generating intial localization results table.")
    raise

global eval
global REPROJECTION_ERROR_THESHOLD_MEAN  # for calculating success rate
global REPROJECTION_ERROR_THESHOLD_STD  # for calculating success rate

eval = InitialLocalizationEvaluation()
REPROJECTION_ERROR_THESHOLD_MEAN = 100
REPROJECTION_ERROR_THESHOLD_STD = 70
RESULTPATH_TO_REFERENCE_MAPPING = {}
controlOpt = {
    "resultsToLoad": [-1],
    "save": False,
    "saveFolder": "data/eval/initialLocalization/plots/accuracyResultTable",
    "saveName": "reprojectionErrors",
    "verbose": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",  # greenscreen
    "data/eval/initialLocalization/results/20230516_113957_Partial",  # greenscreen
    "data/eval/initialLocalization/results/20230516_115857_arena",  # greenscreen
    "data/eval/initialLocalization/results/20230603_143937_modelY",  # assembly board
    "data/eval/initialLocalization/results/20230807_150735_partial",  # assembly board
    "data/eval/initialLocalization/results/20230603_140143_arena",  # assembly board
]
global referenceNames
referenceNames = [
    "modelY_gr",  # modelY + greenscreen
    "partial_gr",  # parial + greenscreen
    "arena_gr",  # arena + greenscreen
    "modelY_ab",  # modelY + assembly board
    "partial_ab",  # parial + assembly board
    "arena_ab",  # arena + assembly board
]

modelNames = [
    "$\\mathcal{T}_{s1}$",  # modelY + greenscreen
    "$\\mathcal{T}_{s2}$",  # parial + greenscreen
    "$\\mathcal{T}_{s1}$",  # arena + greenscreen
    "$\\mathcal{T}_{s1}$",  # modelY + assembly board
    "$\\mathcal{T}_{s2}$",  # parial + assembly board
    "$\\mathcal{T}_{s3}$",  # arena + assembly board
]
# create mappins between results and references
REFERENCE_TO_PATH_MAPPING = {
    ref: path for ref, path in zip(referenceNames, resultFolderPaths)
}
PATH_TO_REFERENCE_MAPPING = {
    path: ref for ref, path in zip(referenceNames, resultFolderPaths)
}
REFERENCE_TO_MODEL_MAPPING = {
    ref: model for ref, model in zip(referenceNames, modelNames)
}


def getReprojectionResultsFromFile_AsDict(resultFolderPath):
    resultFiles = eval.list_result_files(resultFolderPath)
    reprojectionErrorResults = []
    frames = []
    means = []
    stds = []
    for resultFile in resultFiles:
        resultFilePath = os.path.join(resultFolderPath, resultFile)
        result = eval.loadResults(resultFilePath)
        reprojectionErrorResult = eval.calculateReprojectionError(result)
        reprojectionErrorResults.append(reprojectionErrorResult)
        frames.append(result["frame"])
        means.append(reprojectionErrorResult["meanReprojectionError"])
        stds.append(reprojectionErrorResult["stdReprojectionError"])
    frames = np.array(frames)
    means = np.array(means)
    stds = np.array(stds)

    # determine successful frames
    successStates = (means < REPROJECTION_ERROR_THESHOLD_MEAN) & (
        stds < REPROJECTION_ERROR_THESHOLD_STD
    )
    # calucalte success rate
    successRate = successStates.sum() / len(frames) * 100
    # average mean
    averageMean = np.average(means[successStates])
    # average std
    averageStd = np.average(stds[successStates])
    reference = PATH_TO_REFERENCE_MAPPING[resultFolderPath]
    if reference == "modelY_gr" or reference == "partial_gr" or reference == "arena_gr":
        quality = "high"
    else:
        quality = "low"
    resultDict = {
        "resultFolderPath": resultFolderPath,
        "reference": PATH_TO_REFERENCE_MAPPING[resultFolderPath],
        "frames": frames,
        "means": means,
        "stds": stds,
        "successfulFrames": successStates,
        "successRate": successRate,
        "averageMean": averageMean,
        "averageStd": averageStd,
        "quality": quality,
    }
    return resultDict


def printLatexTable(collectionOfResultDicts):
    latex_table = """
    \\begin{tabular}{ccccc}\\toprule
	model		& quality of $\\mathcal{P}$  & num. frames & avg. reprojection error in px & success rate in $\\%$\\\\
	\\midrule
    """
    # order given by reference Name list
    for referenceName in referenceNames:
        # get the corresponding result dict
        resultDict = [
            d for d in collectedResultDicts if d["reference"] == referenceName
        ][0]
        # map reference to model
        model = REFERENCE_TO_MODEL_MAPPING[resultDict["reference"]]
        numFrames = len(resultDict["frames"])
        avg_reprojection_error = resultDict["averageMean"]
        avg_reprojection_error_std = resultDict["averageStd"]
        success_rate = resultDict["successRate"]

        latex_table += f"{model} & {resultDict['quality']} & {numFrames} & {avg_reprojection_error:.1f} \\pm {avg_reprojection_error_std:.1f} & {success_rate:.1f} \\\\\n"

    latex_table += """
    \\bottomrule
    \\end{tabular}
    """

    print(latex_table)


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]
    # load results
    collectedResultDicts = []
    for i, resultFolderPath in enumerate(resultFolderPaths):
        resultDict = getReprojectionResultsFromFile_AsDict(resultFolderPath)
        collectedResultDicts.append(resultDict)
    printLatexTable(collectedResultDicts)
    print("Done.")
