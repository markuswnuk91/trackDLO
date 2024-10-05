import sys
import os
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
except:
    print("Imports for plotting script to plot grasping accuracy tables failed.")
    raise

global eval
eval = GraspingAccuracyEvaluation()

controlOpt = {
    "resultsToLoad": [-1],
    "topologyToEvaluate": "arena",  # modelY, partial, arena
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],
}
styleOpt = {
    "translationalScaleFactor": 100,
    "rotationalScaleFactor": 1,
    "measureRotationIn": "grad",  # "grad", "rad"
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


def multiply_dict_by_value(d, scalar):
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            multiply_dict_by_value(value, scalar)
        else:
            # If the value is not a dictionary (assumed to be a number), multiply by the scalar
            d[key] = value * scalar
    return d


def printTable(translational_error_dict, rotational_error_dict, topology):
    # expected error_dict structure:
    # error_dict[method][str(grasp_index)]["values"]

    # scale values in the dict to set translation to cm and rotation to degree
    translational_error_dict = multiply_dict_by_value(
        translational_error_dict, styleOpt["translationalScaleFactor"]
    )
    rotational_error_dict = multiply_dict_by_value(
        rotational_error_dict, styleOpt["rotationalScaleFactor"]
    )
    # CPD
    # translation
    cpd_0_mean_trans = translational_error_dict["cpd"]["0"]["mean"]
    cpd_1_mean_trans = translational_error_dict["cpd"]["1"]["mean"]
    cpd_2_mean_trans = translational_error_dict["cpd"]["2"]["mean"]
    cpd_3_mean_trans = translational_error_dict["cpd"]["3"]["mean"]
    cpd_0_std_trans = translational_error_dict["cpd"]["0"]["std"]
    cpd_1_std_trans = translational_error_dict["cpd"]["1"]["std"]
    cpd_2_std_trans = translational_error_dict["cpd"]["2"]["std"]
    cpd_3_std_trans = translational_error_dict["cpd"]["3"]["std"]
    # rotation
    cpd_0_mean_rot = rotational_error_dict["cpd"]["0"]["mean"]
    cpd_1_mean_rot = rotational_error_dict["cpd"]["1"]["mean"]
    cpd_2_mean_rot = rotational_error_dict["cpd"]["2"]["mean"]
    cpd_3_mean_rot = rotational_error_dict["cpd"]["3"]["mean"]
    cpd_0_std_rot = rotational_error_dict["cpd"]["0"]["std"]
    cpd_1_std_rot = rotational_error_dict["cpd"]["1"]["std"]
    cpd_2_std_rot = rotational_error_dict["cpd"]["2"]["std"]
    cpd_3_std_rot = rotational_error_dict["cpd"]["3"]["std"]
    # SPR
    # translation
    spr_0_mean_trans = translational_error_dict["spr"]["0"]["mean"]
    spr_1_mean_trans = translational_error_dict["spr"]["1"]["mean"]
    spr_2_mean_trans = translational_error_dict["spr"]["2"]["mean"]
    spr_3_mean_trans = translational_error_dict["spr"]["3"]["mean"]
    spr_0_std_trans = translational_error_dict["spr"]["0"]["std"]
    spr_1_std_trans = translational_error_dict["spr"]["1"]["std"]
    spr_2_std_trans = translational_error_dict["spr"]["2"]["std"]
    spr_3_std_trans = translational_error_dict["spr"]["3"]["std"]
    # rotation
    spr_0_mean_rot = rotational_error_dict["spr"]["0"]["mean"]
    spr_1_mean_rot = rotational_error_dict["spr"]["1"]["mean"]
    spr_2_mean_rot = rotational_error_dict["spr"]["2"]["mean"]
    spr_3_mean_rot = rotational_error_dict["spr"]["3"]["mean"]
    spr_0_std_rot = rotational_error_dict["spr"]["0"]["std"]
    spr_1_std_rot = rotational_error_dict["spr"]["1"]["std"]
    spr_2_std_rot = rotational_error_dict["spr"]["2"]["std"]
    spr_3_std_rot = rotational_error_dict["spr"]["3"]["std"]
    # KPR
    # translation
    kpr_0_mean_trans = translational_error_dict["kpr"]["0"]["mean"]
    kpr_1_mean_trans = translational_error_dict["kpr"]["1"]["mean"]
    kpr_2_mean_trans = translational_error_dict["kpr"]["2"]["mean"]
    kpr_3_mean_trans = translational_error_dict["kpr"]["3"]["mean"]
    kpr_0_std_trans = translational_error_dict["kpr"]["0"]["std"]
    kpr_1_std_trans = translational_error_dict["kpr"]["1"]["std"]
    kpr_2_std_trans = translational_error_dict["kpr"]["2"]["std"]
    kpr_3_std_trans = translational_error_dict["kpr"]["3"]["std"]
    # rotation
    kpr_0_mean_rot = rotational_error_dict["kpr"]["0"]["mean"]
    kpr_1_mean_rot = rotational_error_dict["kpr"]["1"]["mean"]
    kpr_2_mean_rot = rotational_error_dict["kpr"]["2"]["mean"]
    kpr_3_mean_rot = rotational_error_dict["kpr"]["3"]["mean"]
    kpr_0_std_rot = rotational_error_dict["kpr"]["0"]["std"]
    kpr_1_std_rot = rotational_error_dict["kpr"]["1"]["std"]
    kpr_2_std_rot = rotational_error_dict["kpr"]["2"]["std"]
    kpr_3_std_rot = rotational_error_dict["kpr"]["3"]["std"]

    table_head = (
        f"""\\toprule\n"""
        + r"""& \multicolumn{2}{c}{\textbf{grasp 1}}"""
        + r"""& \multicolumn{2}{c}{\textbf{grasp 2}}"""
        + r"""& \multicolumn{2}{c}{\textbf{grasp 3}}"""
        + r"""& \multicolumn{2}{c}{\textbf{grasp 4}}\\\midrule"""
        + f"""\n"""
        + r"""& $e_{\text{trans}}$& $e_{\text{rot}}$"""
        + r"""&$e_{\text{trans}}$& $e_{\text{rot}}$"""
        + r"""&$e_{\text{trans}}$& $e_{\text{rot}}$"""
        + r"""&$e_{\text{trans}}$&  $e_{\text{rot}}$"""
    )

    # table_body_cpd = (
    #     f"""CPD & ${cpd_0_mean_trans:.1f} \\pm {cpd_0_std_trans:.1f}$"""
    #     + f""" & ${cpd_0_mean_rot:.1f} \\pm {cpd_0_std_rot:.1f}$"""
    #     + f""" & ${cpd_1_mean_trans:.1f} \\pm {cpd_1_std_trans:.1f}$"""
    #     + f""" & ${cpd_1_mean_rot:.1f} \\pm {cpd_1_std_rot:.1f}$"""
    #     + f""" & ${cpd_2_mean_trans:.1f} \\pm {cpd_2_std_trans:.1f}$"""
    #     + f""" & ${cpd_2_mean_rot:.1f} \\pm {cpd_2_std_rot:.1f}$"""
    #     + f""" & ${cpd_3_mean_trans:.1f} \\pm {cpd_3_std_trans:.1f}$"""
    #     + f""" & ${cpd_3_mean_rot:.1f} \\pm {cpd_3_std_rot:.1f}$"""
    # )
    # table_body_spr = (
    #     f"""SPR & ${spr_0_mean_trans:.1f} \\pm {spr_0_std_trans:.1f}$"""
    #     + f""" & ${spr_0_mean_rot:.1f} \\pm {spr_0_std_rot:.1f}$"""
    #     + f""" & ${spr_1_mean_trans:.1f} \\pm {spr_1_std_trans:.1f}$"""
    #     + f""" & ${spr_1_mean_rot:.1f} \\pm {spr_1_std_rot:.1f}$"""
    #     + f""" & ${spr_2_mean_trans:.1f} \\pm {spr_2_std_trans:.1f}$"""
    #     + f""" & ${spr_2_mean_rot:.1f} \\pm {spr_2_std_rot:.1f}$"""
    #     + f""" & ${spr_3_mean_trans:.1f} \\pm {spr_3_std_trans:.1f}$"""
    #     + f""" &${spr_3_mean_rot:.1f} \\pm {spr_3_std_rot:.1f}$"""
    # )
    # table_body_kpr = (
    #     f"""KPR & ${kpr_0_mean_trans:.1f} \\pm {kpr_0_std_trans:.1f}$"""
    #     + f""" & ${kpr_0_mean_rot:.1f} \\pm {kpr_0_std_rot:.1f}$"""
    #     + f""" & ${kpr_1_mean_trans:.1f} \\pm {kpr_1_std_trans:.1f}$"""
    #     + f""" & ${kpr_1_mean_rot:.1f} \\pm {kpr_1_std_rot:.1f}$"""
    #     + f""" & ${kpr_2_mean_trans:.1f} \\pm {kpr_2_std_trans:.1f}$"""
    #     + f""" & ${kpr_2_mean_rot:.1f} \\pm {kpr_2_std_rot:.1f}$"""
    #     + f""" & ${kpr_3_mean_trans:.1f} \\pm {kpr_3_std_trans:.1f}$"""
    #     + f""" & ${kpr_3_mean_rot:.1f} \\pm {kpr_3_std_rot:.1f}$"""
    # )
    table_body_cpd = (
        f"""CPD & ${cpd_0_mean_trans:.1f}$"""
        + f""" & ${cpd_0_mean_rot:.1f}$"""
        + f""" & ${cpd_1_mean_trans:.1f}$"""
        + f""" & ${cpd_1_mean_rot:.1f}$"""
        + f""" & ${cpd_2_mean_trans:.1f}$"""
        + f""" & ${cpd_2_mean_rot:.1f}$"""
        + f""" & ${cpd_3_mean_trans:.1f}$"""
        + f""" & ${cpd_3_mean_rot:.1f}$"""
    )
    table_body_spr = (
        f"""SPR & ${spr_0_mean_trans:.1f}$"""
        + f""" & ${spr_0_mean_rot:.1f}$"""
        + f""" & ${spr_1_mean_trans:.1f}$"""
        + f""" & ${spr_1_mean_rot:.1f}$"""
        + f""" & ${spr_2_mean_trans:.1f}$"""
        + f""" & ${spr_2_mean_rot:.1f}$"""
        + f""" & ${spr_3_mean_trans:.1f}$"""
        + f""" &${spr_3_mean_rot:.1f}$"""
    )
    table_body_kpr = (
        f"""KPR & ${kpr_0_mean_trans:.1f}$"""
        + f""" & ${kpr_0_mean_rot:.1f}$"""
        + f""" & ${kpr_1_mean_trans:.1f}$"""
        + f""" & ${kpr_1_mean_rot:.1f}$"""
        + f""" & ${kpr_2_mean_trans:.1f}$"""
        + f""" & ${kpr_2_mean_rot:.1f}$"""
        + f""" & ${kpr_3_mean_trans:.1f}$"""
        + f""" & ${kpr_3_mean_rot:.1f}$"""
    )
    table_end = r"""\bottomrule"""

    # customize depending on topology
    if topology == "modelY":
        table_body = (
            table_body_cpd + "\\\\\n" + table_body_spr + "\\\\\n" + table_body_kpr
        )
    else:
        cpd_4_mean_trans = translational_error_dict["cpd"]["4"]["mean"]
        spr_4_mean_trans = translational_error_dict["spr"]["4"]["mean"]
        kpr_4_mean_trans = translational_error_dict["kpr"]["4"]["mean"]
        cpd_4_mean_rot = rotational_error_dict["cpd"]["4"]["mean"]
        spr_4_mean_rot = rotational_error_dict["spr"]["4"]["mean"]
        kpr_4_mean_rot = rotational_error_dict["kpr"]["4"]["mean"]

        table_head += r"""&$e_{\text{trans}}$&  $e_{\text{rot}}$"""
        table_body_cpd += (
            f""" & ${cpd_4_mean_trans:.1f}$""" + f""" &${cpd_4_mean_rot:.1f}$"""
        )
        table_body_spr += (
            f""" & ${spr_4_mean_trans:.1f}$""" + f""" &${spr_4_mean_rot:.1f}$"""
        )
        table_body_kpr += (
            f""" & ${kpr_4_mean_trans:.1f}$""" + f""" & ${kpr_4_mean_rot:.1f}$"""
        )

        if topology == "partial":
            table_body = (
                table_body_cpd + "\\\\\n" + table_body_spr + "\\\\\n" + table_body_kpr
            )
        else:
            cpd_5_mean_trans = translational_error_dict["cpd"]["5"]["mean"]
            spr_5_mean_trans = translational_error_dict["spr"]["5"]["mean"]
            kpr_5_mean_trans = translational_error_dict["kpr"]["5"]["mean"]
            cpd_5_mean_rot = rotational_error_dict["cpd"]["5"]["mean"]
            spr_5_mean_rot = rotational_error_dict["spr"]["5"]["mean"]
            kpr_5_mean_rot = rotational_error_dict["kpr"]["5"]["mean"]

            table_head += r"""&$e_{\text{trans}}$&  $e_{\text{rot}}$"""
            table_body_cpd += (
                f""" & ${cpd_5_mean_trans:.1f}$""" + f""" & ${cpd_5_mean_rot:.1f}$"""
            )
            table_body_spr += (
                f""" & ${spr_5_mean_trans:.1f}$""" + f""" & ${spr_5_mean_rot:.1f}$"""
            )
            table_body_kpr += (
                f""" & ${kpr_5_mean_trans:.1f}$""" + f""" & ${kpr_5_mean_rot:.1f}$"""
            )
            table_body = (
                table_body_cpd + "\\\\\n" + table_body_spr + "\\\\\n" + table_body_kpr
            )
    table = table_head + "\\\\\n" + table_body + "\\\\\n" + table_end
    print(table)
    return table


def filter_values(
    array_of_values,
    grasps_array,
    methods_array,
    models_array,
    grasp_value,
    method_value,
    model_value,
):
    mask = (
        (grasps_array == grasp_value)
        & (methods_array == method_value)
        & (models_array == model_value)
    )
    filtered_values = array_of_values[mask]
    return filtered_values


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        # load all results of the corresponding topology
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if resultFolderPath.split("_")[-1] == controlOpt["topologyToEvaluate"]
        ]
    else:
        # choose custom results
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
                if styleOpt["measureRotationIn"] == "grad":
                    rotationalGraspingErrors.append(
                        graspingAccuracyError["graspingAngularErrorsInGrad"]
                    )
                elif styleOpt["measureRotationIn"] == "rad":
                    graspingAccuracyError["graspingAngularErrorsInRad"]
                else:
                    raise NotImplementedError
                methods.append(method)
                grasps.append(nRegistrationResult)
                modelName = result["dataSetName"].split("_")[-1]
                models.append(modelName)

    # iterate over grasps
    grasp_indices = set(grasps)
    translational_errors_per_grasp = {}
    rotational_errors_per_grasp = {}
    # convert lists to numpy arrays
    translationalGraspingErrors = np.array(translationalGraspingErrors)
    rotationalGraspingErrors = np.array(rotationalGraspingErrors)
    models = np.array(models)
    methods = np.array(methods)
    grasps = np.array(grasps)
    for method in methodsToEvaluate:
        translational_errors_per_grasp[method] = {}
        rotational_errors_per_grasp[method] = {}
        for grasp_index in grasp_indices:
            translational_errors_per_grasp[method][str(grasp_index)] = {}
            rotational_errors_per_grasp[method][str(grasp_index)] = {}
            values_trans = filter_values(
                array_of_values=translationalGraspingErrors,
                grasps_array=grasps,
                models_array=models,
                methods_array=methods,
                grasp_value=grasp_index,
                method_value=method,
                model_value=controlOpt["topologyToEvaluate"],
            )
            values_rot = filter_values(
                array_of_values=rotationalGraspingErrors,
                grasps_array=grasps,
                models_array=models,
                methods_array=methods,
                grasp_value=grasp_index,
                method_value=method,
                model_value=controlOpt["topologyToEvaluate"],
            )
            translational_errors_per_grasp[method][str(grasp_index)][
                "values"
            ] = values_trans
            translational_errors_per_grasp[method][str(grasp_index)]["mean"] = np.mean(
                values_trans
            )
            translational_errors_per_grasp[method][str(grasp_index)]["std"] = np.std(
                values_trans
            )
            rotational_errors_per_grasp[method][str(grasp_index)]["values"] = values_rot
            rotational_errors_per_grasp[method][str(grasp_index)]["mean"] = np.mean(
                values_rot
            )
            rotational_errors_per_grasp[method][str(grasp_index)]["std"] = np.std(
                values_rot
            )
    printTable(
        translational_error_dict=translational_errors_per_grasp,
        rotational_error_dict=rotational_errors_per_grasp,
        topology=controlOpt["topologyToEvaluate"],
    )
    print("Done.")
