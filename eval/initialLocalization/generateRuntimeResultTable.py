import sys
import os
import numpy as np
import time

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.localization.topologyExtraction.minimalSpanningTreeExtraction import (
        MinimalSpanningTreeExtraction,
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
    "loadResultsFromFile": True,
    "resultsToLoad": [-1],
    "save": True,
    "saveFolder": "data/eval/initialLocalization/runtimes",
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
global dataset_references
global modelNames
dataset_references = [
    "modelY_gr",  # modelY + greenscreen
    "partial_gr",  # parial + greenscreen
    "arena_gr",  # arena + greenscreen
    "modelY_ab",  # modelY + assembly board
    "partial_ab",  # parial + assembly board
    "arena_ab",  # arena + assembly board
]
model_refs_latex = [
    "$\\mathcal{T}_{s1}$",  # modelY
    "$\\mathcal{T}_{s2}$",  # parial
    "$\\mathcal{T}_{s3}$",  # arena
]
model_refs_python = [
    "T1",  # modelY
    "T2",  # parial
    "T3",  # arena
]
PYTONREFERENCE_TO_LATEX_MAPPING = {
    ref: model for ref, model in zip(model_refs_python, model_refs_latex)
}

# create mappins between results and references
REFERENCE_TO_PATH_MAPPING = {
    ref: path for ref, path in zip(dataset_references, resultFolderPaths)
}
PATH_TO_REFERENCE_MAPPING = {
    path: ref for ref, path in zip(dataset_references, resultFolderPaths)
}


def get_runtimes(result):
    """Gathers relevant runtime measurement in a result structure"""

    # preprocessing
    t_prep_start = time.time()
    eval.setConfig(result["config"])
    output_ptCloud = eval.getPointCloud(result["frame"], result["dataSetPath"])
    t_prep_end = time.time()
    input_ptCloud = eval.getPointCloud(
        result["frame"], result["dataSetPath"], segmentationMethod="original"
    )
    preprocessing_results = {
        "total_runtime": t_prep_end - t_prep_start,
        "output_pc_size": len(output_ptCloud[0]),
        "input_pc_size": len(input_ptCloud[0]),
    }
    # lof
    lof_results = {
        "total_runtime": result["lofResult"]["runtime"],
        "input_pc_size": len(result["lofResult"]["outliers"])
        + len(result["lofResult"]["filteredPointSet"]),
        "output_pc_size": len(result["lofResult"]["filteredPointSet"]),
    }

    # l1
    l1_results = {
        "total_runtime": result["l1Result"]["runtimes"]["withoutVisualization"],
        "runtime_per_iteration": result["l1Result"]["runtimes"]["perIteration"],
        "input_pc_size": len(result["l1Result"]["Y"]),
        "output_pc_size": len(result["l1Result"]["X"]),
    }

    # topologyReconstruction
    t_topologyReconstruction_start = time.time()
    mst_reconstruction = MinimalSpanningTreeExtraction(
        result["topologyExtractionResult"]["Y"],
        result["topologyExtractionResult"]["nPaths"],
    )
    t_topologyReconstruction_end = time.time()
    topologyReconstruction_results = {
        "total_runtime": t_topologyReconstruction_end - t_topologyReconstruction_start,
        "input_pc_size": len(result["topologyExtractionResult"]["Y"]),
    }

    # correspondance estimation
    correspondanceEstimation_results = {
        "total_runtime": result["localizationResult"]["runtimes"][
            "correspondanceEstimation"
        ]
    }

    # inverse kinematics
    poseEstimation_results = {
        "total_runtime": result["localizationResult"]["runtimes"][
            "inverseKinematicsWithoutVisualization"
        ],
        "runtime_per_iteration": result["localizationResult"]["runtimes"][
            "inverseKinematicsIterations"
        ],
    }
    result["runtimes"] = {
        "preprocessing": preprocessing_results,
        "l1": l1_results,
        "lof": lof_results,
        "topologyReconstruction": topologyReconstruction_results,
        "correspondanceEstimation": correspondanceEstimation_results,
        "poseEstimation": poseEstimation_results,
    }
    return result


def collect_results_in_dict(resultFolderPath):
    resultFiles = eval.list_result_files(resultFolderPath)
    for resultFile in resultFiles:
        resultFilePath = os.path.join(resultFolderPath, resultFile)
        result = eval.loadResults(resultFilePath)
        runtimeResult = get_runtimes(result)
        result["runtimes"] = runtimeResult
    return result


def query_runtime_data_from_collection(
    collectionOfResults, dataset_keys, processing_step_key
):
    runtimes = []
    for dataset_key in dataset_keys:
        for i, result in enumerate(collectionOfResults[dataset_key]):
            runtimes.append(result["runtimes"][processing_step_key]["total_runtime"])
    query_result = {
        "runtimes": runtimes,
        "mean": np.mean(runtimes),
        "std": np.std(runtimes),
    }
    return query_result


def printLatexTable(collectionOfResults):

    table_data = {}
    processing_steps = [
        "preprocessing",
        "l1",
        "lof",
        "topologyReconstruction",
        "correspondanceEstimation",
        "poseEstimation",
    ]
    # modelY
    table_data["T1"] = {}
    for processing_step in processing_steps:
        table_data["T1"][processing_step] = query_runtime_data_from_collection(
            collectionOfResults,
            dataset_keys=["modelY_gr", "modelY_ab"],
            processing_step_key=processing_step,
        )
    # partial
    table_data["T2"] = {}
    for processing_step in processing_steps:
        table_data["T2"][processing_step] = query_runtime_data_from_collection(
            collectionOfResults,
            dataset_keys=["partial_gr", "partial_ab"],
            processing_step_key=processing_step,
        )
    # arena
    table_data["T3"] = {}
    for processing_step in processing_steps:
        table_data["T3"][processing_step] = query_runtime_data_from_collection(
            collectionOfResults,
            dataset_keys=["arena_gr", "arena_ab"],
            processing_step_key=processing_step,
        )

    latex_table = """
    \\begin{tabular}{ccccccccc}\\toprule
    topology & frames &  $t_{\\text{pre}}$	&  $t_{\\text{skel}}$ & $t_{\\text{rec}}$ & $t_{\\text{crp}}$ & $t_{\\text{pose}}$ & $t_{\\text{total}}$ \\\\
	\\midrule
    """
    for key in list(table_data.keys()):

        model_ref = PYTONREFERENCE_TO_LATEX_MAPPING[key]
        num_frames = len(table_data[key]["preprocessing"]["runtimes"])
        t_pre_mean = np.mean(
            (
                table_data[key]["preprocessing"]["runtimes"]
                + table_data[key]["lof"]["runtimes"]
            )
        )
        t_pre_std = np.std(
            table_data[key]["preprocessing"]["runtimes"]
            + table_data[key]["lof"]["runtimes"]
        )
        t_skel_mean = table_data[key]["l1"]["mean"]
        t_skel_std = table_data[key]["l1"]["std"]

        t_rec_mean = table_data[key]["topologyReconstruction"]["mean"]
        t_rec_std = table_data[key]["topologyReconstruction"]["std"]

        t_corresp_mean = table_data[key]["correspondanceEstimation"]["mean"]
        t_corresp_std = table_data[key]["correspondanceEstimation"]["std"]

        t_pose_mean = table_data[key]["poseEstimation"]["mean"]
        t_pose_std = table_data[key]["poseEstimation"]["std"]

        t_total = np.sum(
            t_pre_mean + t_skel_mean + t_rec_mean + t_corresp_mean + t_pose_mean
        )
        latex_table += f"{model_ref} & {num_frames} & {t_pre_mean:.1f}  &{t_skel_mean:.1f} & {t_rec_mean:.1f} & {t_corresp_mean:.1f} & {t_pose_mean:.1f} & {t_total:.1f}\\\\\n"

    latex_table += """
    \\bottomrule
    \\end{tabular}
    """

    print(latex_table)


if __name__ == "__main__":

    if not controlOpt["loadResultsFromFile"]:
        if controlOpt["resultsToLoad"][0] == -1:
            dataSetsToEvaluate = resultFolderPaths
        else:
            dataSetsToEvaluate = [
                dataSetPath
                for i, dataSetPath in enumerate(resultFolderPaths)
                if i in controlOpt["resultsToLoad"]
            ]
        # load results
        results_all_datasets = {}
        # iterate over datasets
        for i, resultFolderPath in enumerate(dataSetsToEvaluate):
            resultFiles = eval.list_result_files(resultFolderPath)
            # iterate over frames
            runtime_results_dataset = []
            for j, resultFile in enumerate(resultFiles):
                resultFilePath = os.path.join(resultFolderPath, resultFile)
                result = eval.loadResults(resultFilePath)
                result_with_runtimes = get_runtimes(result)
                runtime_results_dataset.append(result_with_runtimes)
                if controlOpt["verbose"]:
                    print(
                        "Collected result for data set:\n{} ({});\nframe: {}".format(
                            i, resultFolderPath, j
                        )
                    )
            results_all_datasets[PATH_TO_REFERENCE_MAPPING[resultFolderPath]] = (
                runtime_results_dataset
            )
            if controlOpt["save"]:
                eval.saveWithPickle(
                    data=results_all_datasets,
                    filePath=os.path.join(
                        controlOpt["saveFolder"], "runtimeResults.pkl"
                    ),
                    recursionLimit=10000,
                )
                if controlOpt["verbose"]:
                    print(
                        "Saved runtime resutls to : {}".format(controlOpt["saveFolder"])
                    )
    elif controlOpt["loadResultsFromFile"]:
        results_all_datasets = eval.loadResults(
            os.path.join(controlOpt["saveFolder"], "runtimeResults.pkl")
        )
    printLatexTable(results_all_datasets)
    print("Done.")
