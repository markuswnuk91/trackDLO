import numpy as np
import dartpy as dart


def linearDeformationJacobian(skel, vGripper, graspNodeIdx, q, k, d):
    # set the current configuration of the skeleton
    skel.setPositions(q)
    # determine jacobians from grasp node to all other nodes
    otherNodeIndices = []
    jacobiansToOtherNodes = []
    for otherNodeIdx in range(0, skel.getNumBodyNodes()):
        if otherNodeIdx is not graspNodeIdx:
            otherNodeIndices.append(otherNodeIdx)
            jacobiansToOtherNodes.append(
                skel.getLinearJacobian(skel.getBodyNode(otherNodeIdx))[
                    :, 6:
                ]  # neglect free joint
            )
    J = np.vstack(jacobiansToOtherNodes)
    # calculate velocity discount based on diminishing rigidity approach according to Berenson, D.: Manipulation of Deformable Objects Without Modeling and Simulating Deformation, IEEE International Conference on Intelligent Robots and Systems (IROS) , 2013
    discountVelocities = []

    for bodyNodeIdx in otherNodeIndices:
        # geodesic Distance
        d_geod = 3
        vDiscount = (1 - np.exp(-k * d_geod)) * vGripper
        discountVelocities.append(vDiscount)
    VDiscount = np.vstack(discountVelocities)

    # make system of equations to project the desired velocity difference on the joint angles with:
    # Jq_dot = V with V stackend vector of discount velocities, q_dot generalized velocities, J stacked Jacobian matrices
    q_dot = dampedPseudoInverse(J, d) @ VDiscount.flatten()

    # reconstruct spatial velcities (which respect kinematic constraints)
    VConstrained = J @ q_dot

    VNonRigid = np.tile(vGripper, ()) - np.reshape(VConstrained, (-1, 3))

    deformationJacobians = []
    for i, bodyNodeIdx in enumerate(otherNodeIndices):
        deformationJacobian = np.diag(vGripper / VNonRigid[i])
        deformationJacobians.append(deformationJacobian)
    J_lindef = np.vstack(deformationJacobians)
    return VNonRigid, J_lindef


def dampedPseudoInverse(J, dampingFactor):
    dim = J.shape[0]
    dampedPseudoInverse = J.T @ np.linalg.inv(
        J @ J.T + dampingFactor**2 * np.eye(dim)
    )
    return dampedPseudoInverse
