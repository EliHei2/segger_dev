from scipy.spatial import Rectangle
import shapely
import numpy as np
import math


class NDTree:
    """
    NDTree is a data structure for recursively splitting multi-dimensional data
    into smaller regions until each leaf node contains less than or equal to a
    specified number of points. It stores these regions in a balanced binary
    tree.

    Attributes
    ----------
    data : np.ndarray
        The input data to be partitioned.
    n : int
        The maximum number of points allowed in a leaf node.
    idx : np.ndarray
        The indices of the input data points.
    boxes : list
        A list to store the bounding boxes (as shapely polygons) of each region
        in the tree.
    rect : Rectangle
        The bounding box of the entire input data space.
    tree : innernode
        The root of the NDTree.
    """

    def __init__(self, data, n):
        """
        Initializes the NDTree with the given data and maximum points per leaf
        node.

        Parameters
        ----------
        data : np.ndarray
            The input data to be partitioned.
        n : int
            The maximum number of points allowed in a leaf node.
        """
        self.data = np.asarray(data)
        self.n = n
        self.idx = np.arange(data.shape[0])
        self.boxes = []
        self.rect = Rectangle(data.min(0), data.max(0))
        self.tree = innernode(self.n, self.idx, self.rect, self)


class innernode:
    """
    Represents a node in the NDTree. Each node either stores a bounding box for
    the data it contains (leaf nodes) or splits the data into two child nodes.

    Attributes
    ----------
    n : int
        The maximum number of points allowed in a leaf node for this subtree.
    idx : np.ndarray
        The indices of the data points in this node.
    tree : NDTree
        The reference to the main NDTree that holds the data and bounding boxes.
    rect : Rectangle
        The bounding box of the data points in this node.
    split_dim : int
        The dimension along which the node splits the data.
    split_point : float
        The value along the split dimension used to divide the data.
    less : innernode
        The child node containing data points less than or equal to the split
        point.
    greater : innernode
        The child node containing data points greater than the split point.
    """

    def __init__(self, n, idx, rect, tree):
        """
        Initializes the innernode and splits the data if necessary.
        """
        self.n = n
        self.idx = idx
        self.tree = tree
        self.rect = rect
        if not n == 1:
            self.split()
        else:
            box = shapely.box(*self.rect.mins, *self.rect.maxes)
            self.tree.boxes.append(box)

    def split(self):
        """
        Recursively splits the node's data into two child nodes along the
        dimension with the largest spread.
        """
        less = math.floor(self.n // 2)
        greater = self.n - less
        data = self.tree.data[self.idx]
        self.split_dim = np.argmax(self.rect.maxes - self.rect.mins)
        data = data[:, self.split_dim]
        self.split_point = np.quantile(data, less / (less + greater))
        mask = data <= self.split_point
        less_rect, greater_rect = self.rect.split(self.split_dim, self.split_point)
        self.less = innernode(less, self.idx[mask], less_rect, self.tree)
        self.greater = innernode(greater, self.idx[~mask], greater_rect, self.tree)
