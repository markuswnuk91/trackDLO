import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.plot.utils.plot3DCurve import (
        plot3DCurve,
    )
    from src.modelling.utils.calculateArcLength import calcArcLengthFromCurveFun
    from src.modelling.curveShapes3D import helixShape
    from src.plot.utils.visualization import (
        visualizePointSets,
        setupVisualizationCallback,
    )
    from src.reconstruction.differentialGeometry.differentialGeometryReconstruction import (
        DifferentialGeometryReconstruction,
    )
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise


if __name__ == "__main__":

    visCallback = setupVisualizationCallback(
        [-3, 3],
        [-3, 3],
        [0, 6],
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail2/",
    )

    # helix definition & reconstuction
    helixCurve = lambda s: helixShape(s, heightScaling=1.0, frequency=2.0)
    arcLenght = calcArcLengthFromCurveFun(helixCurve, 0, 1)
    s = np.linspace(0, 1, 30)
    Sx = s * arcLenght
    Y = helixCurve(s)
    # Y_rot = Y.copy()
    # Y_rot[:, 2] = Y[:, 1]
    # Y_rot[:, 1] = -Y[:, 2]
    #     aPhi = np.array(
    #         [
    #             -1.29759620e00,
    #             -8.44891376e-01,
    #             1.34495649e-01,
    #             4.23887205e-02,
    #             -2.07225584e-03,
    #             1.78590656e-03,
    #             1.50259292e-03,
    #             6.66559857e-04,
    #             5.35499464e-04,
    #             9.14811385e-05,
    #         ]
    #     )
    #     aTheta = np.array(
    #         [
    #             3.87607297e-01,
    #             1.00788649e00,
    #             -2.06659893e-01,
    #             -1.19583051e-01,
    #             -1.09055969e-02,
    #             7.44743546e-04,
    #             -3.38961106e-03,
    #             -3.48546837e-04,
    #             -1.36323544e-03,
    #             -4.78160479e-05,
    #         ]
    #     )
    # aTheta_opt = np.array(
    #     [
    #         -1.43288012,
    #         -0.28178572,
    #         -0.09028775,
    #         1.22789159,
    #         -0.04788957,
    #         -0.00594992,
    #         -0.03840643,
    #         0.12581364,
    #         -0.03074426,
    #         -0.0115702,
    #     ]
    # )
    # aPhi_opt = np.array(
    #     [
    #         -0.00979435,
    #         3.18368101,
    #         -0.6880238,
    #         0.01995769,
    #         0.50629731,
    #         0.01999431,
    #         -0.07103637,
    #         0.01303655,
    #         0.20880257,
    #         0.01202608,
    #     ]
    # )
    continousReconstruction = DifferentialGeometryReconstruction(
        **{
            "Y": Y,
            "Sx": Sx,
            "L": arcLenght,
            "numSc": 30,
            "callback": visCallback,
            "Rtor": 1000,  # use 1000
            "Rflex": 1000,  # use 1000
            "Density": 0,
            "wPosDiff": 10,  # use 10
            #            "aPhi": aPhi,
            #            "aTheta": aTheta,
            "annealingFlex": 0.99,  # use 0.99
            "annealingTor": 0.8,  # use 0.8
        }
    )
    continousReconstruction.estimateShape(numIter=None)

    print("aPhi: {}".format(continousReconstruction.aPhi))
    print("aTheta: {}".format(continousReconstruction.aTheta))
