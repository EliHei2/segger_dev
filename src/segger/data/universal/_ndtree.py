import numpy as np
import shapely


class ND:
    """Represents an axis-aligned bounding box with optional margin."""

    def __init__(self, xmin, ymin, xmax, ymax, margin=0.0):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.margin = margin

    @property
    def bounds(self):
        """Returns bounds with margin applied."""
        return (
            self.xmin - self.margin,
            self.ymin - self.margin,
            self.xmax + self.margin,
            self.ymax + self.margin,
        )

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
                divisions.append(ND(xmin, ymin, xmax, ymax, margin=self.margin))
        return divisions


class NDTree:
    """A spatial tree that partitions data points into rectangular leaves."""

    def __init__(self, data, max_size, margin=0.0):
        self.data = np.asarray(data)
        self.idx = np.arange(data.shape[0])
        self.leaves = []
        self.sizes = []
        self.rect = ND(*data.min(0), *data.max(0), margin=margin)
        self.tree = InnerNode(self.idx, self.rect, self, max_size)


class InnerNode:
    """Recursive tree node that splits data until reaching max leaf size."""

    def __init__(self, idx, rect, tree, max_size):
        self.idx = idx
        self.tree = tree
        self.rect = rect
        self.max_size = max_size

        if len(self.idx) > self.max_size:
            self.split()
        elif len(self.idx) > 0:
            box = shapely.box(*self.rect.bounds)
            self.tree.leaves.append(box)
            self.tree.sizes.append(len(self.idx))

    def split(self):
        data = self.tree.data[self.idx]
        for rect in self.rect.get_divisions():
            xmin, ymin, xmax, ymax = rect.bounds
            mask = (data[:, 0] >= xmin) & (data[:, 0] < xmax)
            mask &= (data[:, 1] >= ymin) & (data[:, 1] < ymax)
            InnerNode(self.idx[mask], rect, self.tree, self.max_size)
