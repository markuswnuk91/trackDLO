# Python program to read
# json file
import json
import numpy as np

path = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/src/evaluation/bdloDesciptions/arena/"
# Opening JSON file
f = open(path + "model.json")

# returns JSON object as
# a dictionary
data = json.load(f)

# Closing file
f.close()

# Iterating through the json
# list
topologyModel = np.array(data["topologyModel"])
branchSpecificationDict = data["branchSpecifications"]
mass = data["mass"]
targetDict = data["targets"]
branchSpecs = []
for key in branchSpecificationDict:
    branchSpecs.append(branchSpecificationDict[key])

targets = []
for key in targetDict:
    targets.append(targetDict[key])

print("Topology model: {}".format(topologyModel))

print("Branch specifications: {}".format(branchSpecs))

print("Mass: {}".format(mass))

print("Targets: {}".format(targets))
