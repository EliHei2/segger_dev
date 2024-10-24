import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rtree.index
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm


def vector_angle(v1, v2):
    # Calculate the dot product and magnitudes of vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Avoid division by zero, clip the cosine values to [-1, 1] for numerical sta lity
    cos_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)

    # Return the angle in degrees
    return np.degrees(np.arccos(cos_angle))


def triangle_angles_from_points(points, triangles):
    angles_list = []

    for tri in triangles:
        # Extract the points based on the triangle's indices
        p1, p2, p3 = points[tri]

        # Define vectors for the triangle sides
        v1 = p2 - p1  # Vector from p1 to p2
        v2 = p3 - p1  # Vector from p1 to p3
        v3 = p3 - p2  # Vector from p2 to p3

        # Calculate the angles using the vectors
        a = vector_angle(v1, v2)  # Angle at vertex p1
        b = vector_angle(-v1, v3)  # Angle at vertex p2
        c = vector_angle(-v2, -v3)  # Angle at vertex p3 (fixed calculation)
        angles_list.append((a, b, c))

    return np.array(angles_list)


def dfs(v, graph, path, colors):
    colors[v] = 1
    path.append(v)
    for d in graph[v]:
        if colors[d] == 0:
            dfs(d, graph, path, colors)


def plot_points(points, s=3, color="black", zorder=None):
    plt.scatter(points[:, 0], points[:, 1], color=color, s=s, zorder=zorder)


def plot_edges(edges, d_max, part=1):
    if part == 1:
        for edge in edges:
            coords = edges[edge]["coords"]
            if len(edges[edge]["simplices"]) < 2:
                color = "magenta"
                if edges[edge]["length"] > 2 * d_max:
                    color = "red"
                # if edges[edge]['simplices'].values()[0]
            else:
                color = "cyan"

            plt.plot(coords[:, 0], coords[:, 1], color=color)

    if part == 2:
        for edge in edges:
            coords = edges[edge]["coords"]
            if len(edges[edge]["simplices"]) < 2:
                color = "magenta"
                if edges[edge]["length"] > 2 * d_max:
                    color = "red"
                # if edges[edge]['simplices'].values()[0]
            else:
                color = "cyan"
            # print(coords)
            plt.plot(coords[:, 0], coords[:, 1], color=color)


class BoundaryIdentification:

    def __init__(self, data):  # 2d d = Delaunay(t[['x_location', 'y_location']].values)
        self.graph = None
        self.edges = {}
        self.d = Delaunay(data)
        self.d_max = self.calculate_d_max(self.d.points)

        # self.angles = triangle_angles_from_points(d.points, d.simplices)
        self.generate_edges()

    def generate_edges(self):
        d = self.d

        edges = {}
        angles = triangle_angles_from_points(d.points, d.simplices)
        for index, simplex in enumerate(d.simplices):
            for p in range(3):
                edge = tuple(sorted((simplex[p], simplex[(p + 1) % 3])))
                if edge not in edges:
                    edges[edge] = {
                        "simplices": {},  # simplex -> angle
                        # "angles": []
                    }

                edges[edge]["simplices"][index] = angles[index][(p + 2) % 3]

        edges_coordinates = d.points[np.array(list(edges.keys()))]
        edges_length = (
            (edges_coordinates[:, 1, 0] - edges_coordinates[:, 0, 0]) ** 2
            + (edges_coordinates[:, 1, 1] - edges_coordinates[:, 0, 1]) ** 2
        ) ** 0.5

        for edge, coords, length in zip(edges, edges_coordinates, edges_length):
            edges[edge]["coords"] = coords
            edges[edge]["length"] = length

        self.edges = edges

    def calculate_part_1(self, plot=True):
        edges = self.edges
        d = self.d
        d_max = self.d_max

        boundary_edges = [edge for edge in edges if len(edges[edge]["simplices"]) < 2]

        if plot:
            plt.figure(figsize=(10, 10))

        iters = 0
        flag = True
        while flag:
            flag = False
            next_boundary_edges = []

            iters += 1
            if plot:
                plt.subplot(330 + iters)
                self.plot(title=f"iteration: {iters}")

            for current_edge in boundary_edges:
                if current_edge not in edges:  # yeah, it changes
                    continue

                if edges[current_edge]["length"] > 2 * d_max:
                    if len(edges[current_edge]["simplices"].keys()) == 0:
                        del edges[current_edge]
                        continue

                    simplex_id = list(edges[current_edge]["simplices"].keys())[0]
                    simplex = d.simplices[simplex_id]

                    # delete edge and the simplex start
                    for edge in self.get_edges_from_simplex(simplex):
                        if edge != current_edge:
                            edges[edge]["simplices"].pop(simplex_id)
                            next_boundary_edges.append(edge)

                    del edges[current_edge]
                    flag = True
                    # delete edge and the simple end

                else:
                    next_boundary_edges.append(current_edge)

            boundary_edges = next_boundary_edges

        if plot:
            plt.subplot(331 + iters)
            self.plot(title="final")
            plt.tight_layout()

    def plot(self, title="", s=3):

        plt.title(title)
        for edge in self.edges:
            coords = self.edges[edge]["coords"]
            if len(self.edges[edge]["simplices"]) < 2:
                color = "magenta"
                if self.edges[edge]["length"] > 2 * self.d_max:
                    color = "red"
            else:
                color = "cyan"
            plt.plot(coords[:, 0], coords[:, 1], color=color)

        plt.scatter(self.d.points[:, 0], self.d.points[:, 1], color="black", s=s)
        plt.axis("equal")
        plt.axis("off")

    def calculate_part_2(self, plot=True):
        edges = self.edges
        d = self.d
        d_max = self.d_max

        boundary_edges = [edge for edge in edges if len(edges[edge]["simplices"]) < 2]
        boundary_edges_length = len(boundary_edges)
        next_boundary_edges = []

        if plot:
            plt.figure(figsize=(10, 10))

        iters = 0
        while len(next_boundary_edges) != boundary_edges_length:
            next_boundary_edges = []

            iters += 1
            if plot:
                plt.subplot(330 + iters)
                self.plot(title=f"iteration: {iters}")

            for current_edge in boundary_edges:
                if current_edge not in edges:  # yeah, it changes
                    continue

                # need to think about!
                if len(edges[current_edge]["simplices"].keys()) == 0:
                    del edges[current_edge]
                    continue

                simplex_id = list(edges[current_edge]["simplices"].keys())[0]
                simplex = d.simplices[simplex_id]
                if (
                    edges[current_edge]["length"] > 1.5 * d_max and edges[current_edge]["simplices"][simplex_id] > 90
                ) or edges[current_edge]["simplices"][simplex_id] > 180 - 180 / 16:

                    # delete edge and the simplex start
                    for edge in self.get_edges_from_simplex(simplex):
                        if edge != current_edge:
                            edges[edge]["simplices"].pop(simplex_id)
                            next_boundary_edges.append(edge)

                    del edges[current_edge]
                    # delete edge and the simple end

                else:
                    next_boundary_edges.append(current_edge)

            boundary_edges_length = len(boundary_edges)
            boundary_edges = next_boundary_edges

        if plot:
            plt.subplot(331 + iters)
            self.plot(title="final")
            plt.tight_layout()

    def calculate_part_3(self):  # inside boundary hole identification
        # TODO
        pass

    def find_cycles(self):
        e = self.edges
        boundary_edges = [edge for edge in e if len(e[edge]["simplices"]) < 2]
        self.graph = self.generate_graph(boundary_edges)
        cycles = self.get_cycles(self.graph)
        try:
            if len(cycles) == 1:
                geom = Polygon(self.d.points[cycles[0]])
            else:
                geom = MultiPolygon([Polygon(self.d.points[c]) for c in cycles if len(c) >= 3])
        except Exception as e:
            print(e, cycles)
            return None

        return geom

    @staticmethod
    def calculate_d_max(points):
        index = rtree.index.Index()
        for i, p in enumerate(points):
            index.insert(i, p[[0, 1, 0, 1]])

        short_edges = []
        for i, p in enumerate(points):
            res = list(index.nearest(p[[0, 1, 0, 1]], 2))[-1]
            short_edges.append([i, res])

        nearest_points = points[short_edges]

        nearest_dists = (
            (nearest_points[:, 0, 0] - nearest_points[:, 1, 0]) ** 2
            + (nearest_points[:, 0, 1] - nearest_points[:, 1, 1]) ** 2
        ) ** 0.5
        d_max = nearest_dists.max()

        return d_max

    @staticmethod
    def get_edges_from_simplex(simplex):
        edges = []
        for p in range(3):
            edges.append(tuple(sorted((simplex[p], simplex[(p + 1) % 3]))))

        return edges

    @staticmethod
    def generate_graph(edges):
        vertices = set([])
        for edge in edges:
            vertices.add(edge[0])
            vertices.add(edge[1])

        vertices = sorted(list(vertices))
        graph = {v: [] for v in vertices}

        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        return graph

    @staticmethod
    def get_cycles(graph: dict):
        colors = {v: 0 for v in graph}
        cycles = []

        for v in graph.keys():
            if colors[v] == 0:
                cycle = []
                dfs(v, graph, cycle, colors)
                cycles.append(cycle)

        return cycles


def generate_boundaries(df, x="x_location", y="y_location", cell_id="segger_cell_id"):
    res = []
    group_df = df.groupby(cell_id)
    for cell_id, t in tqdm(group_df, total=group_df.ngroups):
        res.append({"cell_id": cell_id, "length": len(t), "geom": generate_boundary(t, x=x, y=y)})

    return gpd.GeoDataFrame(
        data=[[b["cell_id"], b["length"]] for b in res],
        geometry=[b["geom"] for b in res],
        columns=["cell_id", "length"],
    )


def generate_boundary(t, x="x_location", y="y_location"):
    if len(t) < 3:
        return None

    bi = BoundaryIdentification(t[[x, y]].values)
    bi.calculate_part_1(plot=False)
    bi.calculate_part_2(plot=False)
    geom = bi.find_cycles()

    return geom


if __name__ == "__main__":
    points = np.array(
        [
            [0, 0],  # Point 0
            [3, 0],  # Point 1
            [0, 4],  # Point 2
            [5, 5],  # Point 3
            [1, 6],  # Point 4
        ]
    )

    simplices = triangles = np.array(
        [
            [0, 1, 2],  # Triangle formed by points 0, 1, 2
            [1, 3, 4],  # Triangle formed by points 1, 3, 4
        ]
    )

    angles = triangle_angles_from_points(points, triangles)
    print("Angles of each triangle (in degrees):")
    print(angles)
