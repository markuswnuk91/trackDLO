def readDataFromPLY(path):
    pointSet = readPointCloudFromPLY(path)[:: int((1 / downsamplingInputRatio)), :3]
    if visControl["visualizeInput"]:
        fig, ax = setupVisualization(pointSet.shape[1])
        plotPointSet(ax=ax, X=pointSet)
        set_axes_equal(ax)
        plt.show(block=False)
    return pointSet
