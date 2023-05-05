# Python program to read
# json file
import json

path = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/src/evaluation/bdloDesciptions/arena/"
# Opening JSON file
f = open(path + "model.json")

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
for i in data["emp_details"]:
    print(i)

# Closing file
f.close()
