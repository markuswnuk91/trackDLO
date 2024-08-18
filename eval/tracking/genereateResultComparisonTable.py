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
    "resultsToLoad": [0, 1, 2],  # 0: modelY, 1: partial, 2: arena
    "methods": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
    "trackingErrorUnitConversionFactor": 100,  # tracking errors in cm
    "geometricErrorUnitConversionFactor": 100,  # geometric errors in cm
    "runtimeUnitConversionFactor": 1000,  # runtimes in ms
    "successRateUnitConversionFactor": 100,  # success rates in %
}

resultFileName = "result.pkl"

resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]


def printTable(tableValueDict):

    header = """----------Cut-------------\n Topology  & Method & $n_{\\text{frames}}$ & $\\nicefrac{\\bar{e}_{\\text{track}}}{\\si{cm}}$ & $\\nicefrac{\\bar{e}_{\\text{reproj}}}{\\si{px}}$ & $\\nicefrac{\\bar{e}_{\\text{geo}}}{\\si{cm}}$ & $\\nicefrac{r_{\\text{success}}}{\\si{cm}}$ & $\\nicefrac{\\bar{t}_{\\text{runtime}}}{\\si{\\milli\\second}}$\\\\\\midrule\n"""

    table_section_topology_1 = (
        """\\multirow{3}{*}{$1$} & \\acs{CPD} & \\multirow{3}{*}{$695$}"""
        + (
            f"""& ${tableValueDict["topology_1"]["cpd"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["cpd"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_1"]["cpd"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["cpd"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_1"]["cpd"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["cpd"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_1"]["cpd"]["successRate"]:.1f}$ & $ {tableValueDict["topology_1"]["cpd"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_1"]["cpd"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{SPR} & """
        + (
            f"""& ${tableValueDict["topology_1"]["spr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["spr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_1"]["spr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["spr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_1"]["spr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["spr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_1"]["spr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_1"]["spr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_1"]["spr"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{KPR} & """
        + f"""& ${tableValueDict["topology_1"]["kpr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["kpr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_1"]["kpr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["kpr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_1"]["kpr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_1"]["kpr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_1"]["kpr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_1"]["kpr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_1"]["kpr"]["runtime_std"]:.1f}$\\\\\\midrule\n"""
    )

    table_section_topology_2 = (
        """\\multirow{3}{*}{$2$} & \\acs{CPD} & \\multirow{3}{*}{$315$}"""
        + (
            f"""& ${tableValueDict["topology_2"]["cpd"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["cpd"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_2"]["cpd"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["cpd"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_2"]["cpd"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["cpd"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_2"]["cpd"]["successRate"]:.1f}$ & $ {tableValueDict["topology_2"]["cpd"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_2"]["cpd"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{SPR} & """
        + (
            f"""& ${tableValueDict["topology_2"]["spr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["spr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_2"]["spr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["spr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_2"]["spr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["spr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_2"]["spr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_2"]["spr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_2"]["spr"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{KPR} & """
        + f"""& ${tableValueDict["topology_2"]["kpr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["kpr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_2"]["kpr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["kpr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_2"]["kpr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_2"]["kpr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_2"]["kpr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_2"]["kpr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_2"]["kpr"]["runtime_std"]:.1f}$\\\\\\midrule\n"""
    )

    table_section_topology_3 = (
        """\\multirow{3}{*}{$3$} & \\acs{CPD} & \\multirow{3}{*}{$497$}"""
        + (
            f"""& ${tableValueDict["topology_3"]["cpd"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["cpd"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_3"]["cpd"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["cpd"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_3"]["cpd"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["cpd"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_3"]["cpd"]["successRate"]:.1f}$ & $ {tableValueDict["topology_3"]["cpd"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_3"]["cpd"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{SPR} & """
        + (
            f"""& ${tableValueDict["topology_3"]["spr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["spr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_3"]["spr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["spr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_3"]["spr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["spr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_3"]["spr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_3"]["spr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_2"]["spr"]["runtime_std"]:.1f}$\\\\\n"""
        )
        + """& \\acs{KPR} & """
        + f"""& ${tableValueDict["topology_3"]["kpr"]["trackingError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["kpr"]["trackingError_std"]:.1f}$ & ${tableValueDict["topology_3"]["kpr"]["reprojectionError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["kpr"]["reprojectionError_std"]:.1f}$ & ${tableValueDict["topology_3"]["kpr"]["geometricError_mean"]:.1f} \\pm {tableValueDict["topology_3"]["kpr"]["geometricError_std"]:.1f}$ & ${tableValueDict["topology_3"]["kpr"]["successRate"]:.1f}$ & $ {tableValueDict["topology_3"]["kpr"]["runtime_mean"]:.1f} \\pm {tableValueDict["topology_3"]["kpr"]["runtime_std"]:.1f}$\\\\\n"""
    )
    closure = """\\bottomrule"""
    table = (
        header
        + table_section_topology_1
        + table_section_topology_2
        + table_section_topology_3
        + closure
    )
    print(table)
    return


def loadResult(filePath):
    _, file_extension = os.path.splitext(filePath)
    if file_extension == ".pkl":
        with open(filePath, "rb") as f:
            result = pickle.load(f)
    return result


if __name__ == "__main__":
    # table values
    tableValues = {
        "topology_1": {
            "n_frames": 0,
            "cpd": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "spr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "kpr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
        },
        "topology_2": {
            "n_frames": 0,
            "cpd": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "spr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "kpr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
        },
        "topology_3": {
            "n_frames": 0,
            "cpd": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "spr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
            "kpr": {
                "trackingError_mean": 0,
                "trackingError_std": 0,
                "reprojectionError_mean": 0,
                "reprojectionError_std": 0,
                "geometricError_mean": 0,
                "geometricError_std": 0,
                "successRate": 0,
                "runtime_mean": 0,
                "runtime_std": 0,
            },
        },
    }
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)

    # create plot
    for i, result in enumerate(results):
        topologyRef = "topology_" + str(i + 1)
        tableValues[topologyRef]["n_frames"] = len(
            result["trackingResults"][controlOpt["methods"][0]]["frames"]
        )
        for method in controlOpt["methods"]:
            # 1) extract evaluation values from results
            trackingErrors = eval.calculateTrackingErrors(
                result["trackingResults"][method]
            )
            reprojectionErrors = eval.calculateReprojectionErrors(
                result["trackingResults"][method]
            )
            geometricErrors = eval.calculateGeometricErrors(
                result["trackingResults"][method]
            )
            runtimes = eval.calculateRuntimes(result["trackingResults"][method])
            # determine success rate
            successRates = eval.calculateSuccessRate(result["trackingResults"][method])

            # 2) write table entry dict
            # tracking error
            tableValues[topologyRef][method]["trackingError_mean"] = (
                np.mean(trackingErrors)
                * controlOpt["trackingErrorUnitConversionFactor"]
            )
            tableValues[topologyRef][method]["trackingError_std"] = (
                np.std(trackingErrors) * controlOpt["trackingErrorUnitConversionFactor"]
            )
            # reporjection error
            tableValues[topologyRef][method]["reprojectionError_mean"] = np.mean(
                reprojectionErrors["means"]
            )
            tableValues[topologyRef][method]["reprojectionError_std"] = np.mean(
                reprojectionErrors["stds"]
            )
            # geometric error
            tableValues[topologyRef][method]["geometricError_mean"] = (
                np.mean(geometricErrors["lengthError"])
                * controlOpt["geometricErrorUnitConversionFactor"]
            )
            tableValues[topologyRef][method]["geometricError_std"] = (
                np.std(geometricErrors["lengthError"])
                * controlOpt["geometricErrorUnitConversionFactor"]
            )
            # success rate
            tableValues[topologyRef][method]["successRate"] = (
                successRates["successRate"]
                * controlOpt["successRateUnitConversionFactor"]
            )
            # runtimes
            tableValues[topologyRef][method]["runtime_mean"] = (
                np.mean(runtimes["runtimesPerIteration"])
                * controlOpt["runtimeUnitConversionFactor"]
            )
            tableValues[topologyRef][method]["runtime_std"] = (
                np.std(runtimes["runtimesPerIteration"])
                * controlOpt["runtimeUnitConversionFactor"]
            )
    printTable(tableValues)
