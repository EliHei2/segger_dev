from scipy.spatial import Rectangle
import shapely
import numpy as np
import math

class NDTree():

    def __init__(self, data, n):
        
        self.data = np.asarray(data)
        self.n = n
        self.idx = np.arange(data.shape[0])
        self.boxes = []
        self.rect = Rectangle(data.min(0), data.max(0))
        self.tree = innernode(self.n, self.idx, self.rect, self)

class innernode():
    
    def __init__(self, n, idx, rect, tree):
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
        less = math.floor(self.n // 2)
        greater = self.n - less
        data = self.tree.data[self.idx]
        self.split_dim = np.argmax(self.rect.maxes - self.rect.mins)
        data = data[:, self.split_dim]
        self.split_point = np.quantile(data, less / (less + greater))
        mask = data <= self.split_point
        less_rect, greater_rect = self.rect.split(
            self.split_dim,
            self.split_point
        )
        self.less = innernode(
            less,
            self.idx[mask],
            less_rect,
            self.tree
        )
        self.greater = innernode(
            greater,
            self.idx[~mask],
            greater_rect,
            self.tree
        )