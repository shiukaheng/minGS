class BasePointCloud:
    """
    Base point cloud class used for initializing Gaussian positions and colors.
    """
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors