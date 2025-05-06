from scipy.spatial import Rectangle
import shapely
import numpy as np
import math


class ND:
    """Represents an axis-aligned bounding box."""

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def bounds(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_divisions(self):
        """Splits the rectangle into 2xN or Nx2 subdivisions, optimizing for near-square aspect ratio."""
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin

        if w >= h:
            nrows = 2
            ncols = np.rint(w / h * 2).astype(int)
        else:
            ncols = 2
            nrows = np.rint(h / w * 2).astype(int)

        dx = w / ncols
        dy = h / nrows
        divisions = []
        for i in range(ncols):
            for j in range(nrows):
                xmin = self.xmin + i * dx
                ymin = self.ymin + j * dy
                xmax = xmin + dx
                ymax = ymin + dy
                divisions.append(ND(xmin, ymin, xmax, ymax))
        return divisions
        


class NDTree:
    #TODO: Add documentation
    
    def __init__(self, data, max_size):
        #TODO: Add documentation
        
        self.data = np.asarray(data)
        self.idx = np.arange(data.shape[0])
        self.leaves = []
        self.rect = ND(*data.min(0), *data.max(0))
        self.tree = innernode(self.idx, self.rect, self, max_size)


class innernode:
    #TODO: Add documentation

    def __init__(self, idx, rect, tree, max_size):
        #TODO: Add documentation

        self.idx = idx
        self.tree = tree
        self.rect = rect
        self.max_size = max_size
        if len(self.idx) > self.max_size:
            self.split()
        elif len(self.idx) > 0:
            box = shapely.box(*self.rect.bounds)
            self.tree.leaves.append(box)

    def split(self):
        #TODO: Add documentation
        data = self.tree.data[self.idx]
        for rect in self.rect.get_divisions():
            xmin, ymin, xmax, ymax = rect.bounds
            mask = (data[:, 0] >= xmin) & (data[:, 0] < xmax)
            mask &= (data[:, 1] >= ymin) & (data[:, 1] < ymax)
            innernode(self.idx[mask], rect, self.tree, self.max_size)

