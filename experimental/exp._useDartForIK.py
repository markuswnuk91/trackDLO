# get the IK for a bodyNode
inverseKinematics = skel.getBodyNode(0).getOrCreateIK()
# get the current target
targetFrame = inverseKinematics.getTarget()
# set a new target
newTargetFrame = targetFrame.clone()
desiredPosition = np.array(([1, 1, 1]))
newTargetFrame.setTranslation(desiredPosition)
inverseKinematics.setTarget(newTargetFrame)
# solveIK
inverseKinematics.solveAndApply()
# retrieve the solution
q = skel.getPositions()
