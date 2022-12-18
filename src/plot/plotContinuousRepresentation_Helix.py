import json
import matplotlib.pyplot as plt

dataPath = "src/plot/plotdata/helixExample"

# load parameters
with open(dataPath + ".json", "r") as fp:
    parameters = json.load(fp)

X = continuousDLO.getPositions()
