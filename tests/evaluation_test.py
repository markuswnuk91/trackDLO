import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.evaluation.evaluation import Evaluation
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
except:
    print("Imports for Evalutation tests failed.")
    raise


def testEvaluation():
    evaluation = Evaluation(
        pathToConfigFile="tests/testdata/evaluation/evalConfig.json"
    )
    print(evaluation.evalConfig)


def testTrackingEvaluation():
    evaluation = TrackingEvaluation(
        pathToConfigFile="tests/testdata/evaluation/evalConfig.json"
    )
    print(evaluation.evalConfig)


if __name__ == "__main__":
    testEvaluation()
    testTrackingEvaluation()
