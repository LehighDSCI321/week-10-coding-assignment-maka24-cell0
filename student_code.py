
"""This module contains the VersatileDigraph, SortableDigraph, and TraversableDigraph classes."""

from importlib import import_module
from collections import deque

class VersatileDigraph:
    """This is the versatile digraph class."""

    def __init__(self):
        self.__edge_weights = {}
        self.__edge_names = {}
        self.__edge_head = {}
        self.__node_values = {}

    def add_edge(self, tail, head, **vararg):
        """Add an edge to the graph."""
        if not isinstance(tail, str) or not isinstance(head, str):
            raise TypeError("Head and tail must be strings.")
        if tail not in self.get_nodes():
            self.add_node(tail, vararg.get("start_node_value", 0))
        if head not in self.get_nodes():
            self.add_node(head, vararg.get("end_node_value", 0))

        edge_name = vararg.get("name", f"{tail} to {head}")

        # map tail -> head -> edge_name
        self.__edge_names[tail][head] = edge_name
        # map tail -> edge_name -> head (for lookup by name)
        self.__edge_head[tail][edge_name] = head

        weight = (
            vararg.get("edge_weight",
                       vararg.get("weight", 0))
        )
        if weight >= 0:
            self.__edge_weights[tail][head] = weight
        else:
            raise ValueError("Edge weight must be non-negative.")

    def get_nodes(self):
        """Return an iterable of node IDs in the graph."""
        return self.__node_values.keys()

    def add_node(self, node_id, node_value=0):
        """Add a node to the graph."""
        if not isinstance(node_id, str) or not isinstance(node_value, (float, int)):
            raise TypeError(
                "Node ID must be a string and node value must be numeric."
            )
        self.__node_values[node_id] = node_value
        self.__edge_weights[node_id] = {}
        self.__edge_names[node_id] = {}
        self.__edge_head[node_id] = {}

    def get_edge_weight(self, tail, head):
        """Return the weight of an edge."""
        if tail not in self.get_nodes():
            raise KeyError(f"Node {tail} is not present in the graph.")
        if head not in self.get_nodes():
            raise KeyError(f"Node {head} is not present in the graph.")
        if head not in self.__edge_weights[tail]:
            raise KeyError("The specified edge does not exist in the graph.")
        return self.__edge_weights[tail][head]

    def get_node_value(self, node):
        """Return the value of a node."""
        if node not in self.get_nodes():
            raise KeyError(f"Node {node} is not present in the graph.")
        return self.__node_values[node]

    def print_graph(self):
        """Print sentences describing the graph."""
        for tail in self.get_nodes():
            tail_val = self.get_node_value(tail)
            print(f"Node {tail} has a value of {tail_val}.")
            for head in self.__edge_weights[tail]:
                weight = self.get_edge_weight(tail, head)
                name = self.__edge_names[tail][head]
                print(
                    "There is an edge from node "
                    f"{tail} to node {head} of weight {weight} and name {name}."
                )

    def predecessors(self, node):
        """Return a list of the predecessors of a node."""
        if node not in self.get_nodes():
            raise KeyError(f"Node {node} is not present in the graph.")
        return [n for n in self.get_nodes() if node in self.__edge_names[n]]

    def successors(self, node):
        """Return a list of the successors of a node."""
        if node not in self.get_nodes():
            raise KeyError(f"Node {node} is not present in the graph.")
        return list(self.__edge_names[node].keys())

    def in_degree(self, node):
        """Return the in-degree of a node."""
        if node not in self.get_nodes():
            raise KeyError(f"Node {node} is not present in the graph.")
        return len(self.predecessors(node))

    def out_degree(self, node):
        """Return the out-degree of a node."""
        if node not in self.get_nodes():
            raise KeyError(f"Node {node} is not present in the graph.")
        return len(self.successors(node))

    def successor_on_edge(self, tail, edge_name):
        """
        Given a node and an edge name, return the successor of the node
        on that named edge.
        """
        if tail not in self.get_nodes():
            raise KeyError(f"Node {tail} is not present in the graph.")
        if edge_name not in self.__edge_head[tail]:
            raise KeyError(
                f"There is no edge {edge_name} associated with node {tail}"
            )
        return self.__edge_head[tail][edge_name]

    def plot_graph(self):
        """Plot the graph using graphviz (imported dynamically)."""
        try:
            graphviz = import_module("graphviz")
        except Exception as exc:  # ImportError or environment issues
            raise RuntimeError(
                "plot_graph requires the 'graphviz' package."
            ) from exc

        gra = graphviz.Digraph()
        for start_node in self.get_nodes():
            for end_node in self.__edge_weights[start_node]:
                label = self.__edge_names[start_node][end_node]
                weight = self.__edge_weights[start_node][end_node]
                start_label = f"{start_node} ({self.__node_values[start_node]})"
                end_label = f"{end_node} ({self.__node_values[end_node]})"
                gra.edge(start_label, end_label, label=f"{label} ({weight})")
        gra.view()

    def plot_edge_weights(self):
        """Make a bar graph showing the weights of edges (imports dynamically)."""
        try:
            bokeh_io = import_module("bokeh.io")
            bokeh_plotting = import_module("bokeh.plotting")
        except Exception as exc:
            raise RuntimeError(
                "plot_edge_weights requires the 'bokeh' package."
            ) from exc

        bokeh_io.output_file("bar.html")

        edge_dict = {}
        for start in self.get_nodes():
            for end in self.__edge_weights[start]:
                nm = self.__edge_names[start][end]
                edge_dict[f"{start}->{end} ({nm})"] = self.__edge_weights[start][end]

        edge_names = list(edge_dict.keys())
        edge_weights = list(edge_dict.values())

        y_range = edge_names if edge_names else []

        fig = bokeh_plotting.figure(
            y_range=y_range,
            height=350,
            title="Edge Weights",
            toolbar_location=None,
            tools="",
        )
        fig.hbar(y=edge_names, right=edge_weights, height=0.9)
        bokeh_io.show(fig)

class SortableDigraph(VersatileDigraph):
    """Extends VersatileDigraph with a topological sort."""

    def top_sort(self):
        """Return a list of node IDs in topological order."""
        nodes = list(self.get_nodes())
        in_deg = {node: self.in_degree(node) for node in nodes}

        zero_queue = [node for node in nodes if in_deg[node] == 0]

        result = []
        while zero_queue:
            node = zero_queue.pop()
            result.append(node)
            for succ in self.successors(node):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    zero_queue.append(succ)

        if len(result) != len(nodes):
            raise ValueError("Graph has a cycle which means sort if not possible.")
        return result

class TraversableDigraph(SortableDigraph):
    """Extends SortableDigraph with traversal methods."""

    def dfs(self, start_node):
        """Depth-first search traversal starting at start_node."""
        if start_node not in self.get_nodes():
            raise KeyError(f"Node {start_node} is not present in the graph.")

        visited = set()

        def visit(u):
            if u in visited:
                return
            visited.add(u)

            if u != start_node:
                yield u

            for v in self.successors(u):
                yield from visit(v)

        yield from visit(start_node)

    def bfs(self, start_node):
        """Breadth-first search traversal starting at start_node."""
        if start_node not in self.get_nodes():
            raise KeyError(f"Node {start_node} is not present in the graph.")

        visited = set()
        queue = deque()
        queue.append(start_node)

        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)

            if u != start_node:
                yield u

            for v in self.successors(u):
                queue.append(v)

class DAG(TraversableDigraph):
    """A Directed Acyclic Graph."""

    def add_edge(self, tail, head, **vararg):
        """Adding edge to DAG class, ensuring no cycles are created."""

        if tail not in self.get_nodes():
            self.add_node(tail, vararg.get("start_node_value", 0))
        if head not in self.get_nodes():
            self.add_node(head, vararg.get("end_node_value", 0))

        if self._reachable(start_node=head, target_node=tail):
            raise ValueError(
                f"Adding edge {tail} -> {head} would create a cycle in the DAG."
            )

        super().add_edge(tail, head, **vararg)

    def _reachable(self, start_node, target_node):
        """Check if target_node is reachable from start_node using BFS."""
        if start_node == target_node:
            return True

        seen = set()
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            if u == target_node:
                return True
            if u in seen:
                continue
            seen.add(u)
            for v in self.successors(u):
                queue.append(v)

        return False
