"""Exercise the production route helpers without importing Habitat."""

import ast
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "vlnce_baselines/models/graph_utils.py"
tree = ast.parse(PATH.read_text())
distance = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "calc_position_distance")
graph = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraphMap")
methods = [n for n in graph.body if isinstance(n, ast.FunctionDef) and n.name in
           ("front_to_ghost_dist", "paths_to_ghost", "path_to_ghost")]
cls = ast.ClassDef(name="RouteGraph", bases=[], keywords=[], body=methods, decorator_list=[])
namespace = {"np": np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[distance, cls], type_ignores=[])),
             str(PATH), "exec"), namespace)
RouteGraph = namespace["RouteGraph"]


class GhostRoutePolicyTest(unittest.TestCase):
    def graph(self):
        graph = RouteGraph()
        graph.ghost_aug_pos = {"g": np.array([0., 0., 0.])}
        graph.node_pos = {"near": np.array([.2, 0., 0.]),
                          "current": np.array([1., 0., 0.])}
        graph.ghost_fronts = {"g": ["near", "current"]}
        graph.shortest_dist = {"current": {"near": 10., "current": 0.}}
        return graph

    def test_legacy_shortest_route_is_preserved(self):
        self.assertEqual(self.graph().path_to_ghost("current", "g"), (1., 1, "current"))

    def test_official_nearest_front_agrees_with_feature_distance(self):
        graph = self.graph()
        route = graph.path_to_ghost("current", "g", route_mode="nearest_front")
        distance, front = graph.front_to_ghost_dist("g")
        self.assertEqual(route, (10.2, 0, "near"))
        self.assertEqual(route[0], graph.shortest_dist["current"][front] + distance)

    def test_unknown_mode_and_disconnected_graph_fail_closed(self):
        graph = self.graph()
        with self.assertRaises(ValueError):
            graph.path_to_ghost("current", "g", route_mode="guess")
        with self.assertRaises(KeyError):
            graph.path_to_ghost("missing", "g")


if __name__ == "__main__":
    unittest.main()
