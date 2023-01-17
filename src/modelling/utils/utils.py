import sys, os
import numpy as np
import json

try:
    sys.path.append(os.getcwd().replace("/modelling/utils", ""))
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
except:
    print("Imports for plotting continuous vs discrete approximations failed.")
    raise


def loadWakamatsuModelFromJson(filePath=None):
    if filePath is None:
        raise ValueError("Expected file path, insted got None")
    else:
        f = open(filePath)
        data = json.load(f)
        f.close()
        continuousModelParams = {}
        model = WakamatsuModel(
            **{
                "aPhi": np.array(data["aPhi"]),
                "aTheta": np.array(data["aTheta"]),
                "aPsi": np.array(data["aPsi"]),
                "L": data["L"],
                "Rtor": data["Rtor"],
                "Rflex": data["Rflex"],
                "Roh": data["Roh"],
                "N": data["N"],
                "x0": np.array(data["x0"]),
            }
        )
        return model
