# Copyright 2017-2019 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy

from .readers.hpctoolkit_reader import HPCToolkitReader
from .readers.caliper_reader import CaliperReader
from .readers.gprof_dot_reader import GprofDotReader
from .node import Node
from .graph import Graph
from .frame import Frame

lit_idx = 0
squ_idx = 0


class GraphFrame:
    """An input dataset is read into an object of this type, which includes a graph
    and a dataframe.
    """

    def __init__(self, graph=None, dataframe=pd.DataFrame()):
        self.graph = graph
        self.dataframe = dataframe

    def from_hpctoolkit(self, dirname):
        """Read in an HPCToolkit database directory."""
        reader = HPCToolkitReader(dirname)

        (
            self.graph,
            self.dataframe,
            self.exc_metrics,
            self.inc_metrics,
        ) = reader.create_graphframe()

    def from_caliper(self, filename, query):
        """Read in a Caliper `cali` file.

        Args:
            filename (str): name of a Caliper output file in `.cali` format.
            query (str): cali-query in CalQL format.
        """
        reader = CaliperReader(filename, query)

        (
            self.graph,
            self.dataframe,
            self.exc_metrics,
            self.inc_metrics,
        ) = reader.create_graphframe()

    def from_caliper_json(self, filename_or_stream):
        """Read in a Caliper `cali-query` JSON-split file or an open file object.

        Args:
            filename_or_stream (str or file-like): name of a Caliper JSON-split
                output file, or an open file object to read one
        """
        reader = CaliperReader(filename_or_stream)

        (
            self.graph,
            self.dataframe,
            self.exc_metrics,
            self.inc_metrics,
        ) = reader.create_graphframe()

    def from_gprof_dot(self, filename):
        """Read in a DOT file generated by gprof2dot."""
        reader = GprofDotReader(filename)

        (
            self.graph,
            self.dataframe,
            self.exc_metrics,
            self.inc_metrics,
        ) = reader.create_graphframe()

    def from_literal(self, graph_dict):
        """Read graph from a list of dicts literal."""

        def parse_node_literal(child_dict, hparent):
            """Create node_dict for one node and then call the function
            recursively on all children.
            """

            hnode = Node(Frame({"name": child_dict["name"]}), hparent)

            node_dicts.append(
                dict(
                    {"node": hnode, "name": child_dict["name"]}, **child_dict["metrics"]
                )
            )
            hparent.add_child(hnode)

            if "children" in child_dict:
                for child in child_dict["children"]:
                    parse_node_literal(child, hnode)

        list_roots = []
        node_dicts = []

        # start with creating a node_dict for each root
        for i in range(len(graph_dict)):
            graph_root = Node(Frame({"name": graph_dict[i]["name"]}), None)

            node_dict = {"node": graph_root, "name": graph_dict[i]["name"]}
            node_dict.update(**graph_dict[i]["metrics"])
            node_dicts.append(node_dict)

            list_roots.append(graph_root)

            # call recursively on all children of root
            if "children" in graph_dict[i]:
                for child in graph_dict[i]["children"]:
                    parse_node_literal(child, graph_root)

        self.exc_metrics = []
        self.inc_metrics = []
        for key in graph_dict[i]["metrics"].keys():
            if "(inc)" in key:
                self.inc_metrics.append(key)
            else:
                self.exc_metrics.append(key)

        self.graph = Graph(list_roots)
        self.dataframe = pd.DataFrame(data=node_dicts)
        self.dataframe.set_index(["node"], drop=False, inplace=True)

    def copy(self):
        """Return a copy of the graphframe."""
        node_clone = {}
        graph_copy = self.graph.copy(node_clone)
        dataframe_copy = self.dataframe.copy()

        dataframe_copy["node"] = dataframe_copy["node"].apply(lambda x: node_clone[x])
        index_names = self.dataframe.index.names
        dataframe_copy.set_index(index_names, inplace=True, drop=False)

        gf_copy = GraphFrame(graph_copy, dataframe_copy)
        gf_copy.exc_metrics = self.exc_metrics
        gf_copy.inc_metrics = self.inc_metrics
        return gf_copy

    def update_inclusive_columns(self):
        """Update inclusive columns (typically after operations that rewire the
        graph.
        """
        for root in self.graph.roots:
            for node in root.traverse(order="post"):
                for metric in self.exc_metrics:
                    val = self.dataframe.loc[node, metric]
                    for child in node.children:
                        val += self.dataframe.loc[child, metric]
                    inc_metric = metric + " (inc)"
                    self.dataframe.loc[node, inc_metric] = val

    def drop_index_levels(self, function=numpy.mean):
        """Drop all index levels but 'node'."""
        index_names = list(self.dataframe.index.names)
        index_names.remove("node")

        # create dict that stores aggregation function for each column
        agg_dict = {}
        for col in self.dataframe.columns.tolist():
            if col in self.exc_metrics + self.inc_metrics:
                agg_dict[col] = function
            else:
                agg_dict[col] = lambda x: x.iloc[0]

        # perform a groupby to merge nodes that just differ in index columns
        self.dataframe.reset_index(level="node", inplace=True, drop=True)
        agg_df = self.dataframe.groupby("node").agg(agg_dict)
        agg_df.drop(index_names, axis=1, inplace=True)

        self.dataframe = agg_df

    def filter(self, filter_function):
        """Filter the dataframe using a user supplied function."""
        filtered_rows = self.dataframe.apply(filter_function, axis=1)
        filtered_df = self.dataframe[filtered_rows]

        filtered_gf = GraphFrame(self.graph, filtered_df)
        filtered_gf.exc_metrics = self.exc_metrics
        filtered_gf.inc_metrics = self.inc_metrics

        return filtered_gf

    def squash(self):
        """Squash the graph after a filtering operation on the graphframe."""
        global squ_idx
        num_nodes = len(self.graph)

        # calculate number of unique nodes in the dataframe
        # and a set of filtered nodes
        if "rank" in self.dataframe.index.names:
            num_rows_df = len(self.dataframe.groupby(["node"]))
            filtered_nodes = self.dataframe.index.levels[0]
        else:
            num_rows_df = len(self.dataframe.index)
            filtered_nodes = self.dataframe.index

        node_clone = {}

        # function to connect a node to the nearest descendants that are in the
        # list of filtered nodes
        def rewire_tree(node, clone, is_root, roots):
            global squ_idx

            cur_children = node.children
            new_children = []

            # iteratively go over the children of a node
            while cur_children:
                for child in cur_children:
                    cur_children.remove(child)
                    if child in filtered_nodes:
                        new_children.append(child)
                    else:
                        for grandchild in child.children:
                            cur_children.append(grandchild)

            label_to_new_child = {}
            if node in filtered_nodes:
                # create new clones for each child in new_children and rewire
                # with this node
                for new_child in new_children:
                    node_label = new_child.frame
                    if node_label not in label_to_new_child.keys():
                        new_child_clone = Node(new_child.frame, clone)
                        squ_idx += 1
                        clone.add_child(new_child_clone)
                        label_to_new_child[node_label] = new_child_clone
                    else:
                        new_child_clone = label_to_new_child[node_label]

                    node_clone[new_child] = new_child_clone
                    rewire_tree(new_child, new_child_clone, False, roots)
            elif is_root:
                # if we reach here, this root is not in the graph anymore
                # make all its nearest descendants roots in the new graph
                for new_child in new_children:
                    new_child_clone = Node(new_child.frame, None)
                    node_clone[new_child] = new_child_clone
                    squ_idx += 1
                    roots.append(new_child_clone)
                    rewire_tree(new_child, new_child_clone, False, roots)

        squ_idx = 0

        new_roots = []
        # only do a squash if a filtering operation has been applied
        if num_nodes != num_rows_df:
            for root in self.graph.roots:
                if root in filtered_nodes:
                    clone = Node(root.frame, None)
                    new_roots.append(clone)
                    node_clone[root] = clone
                    squ_idx += 1
                    rewire_tree(root, clone, True, new_roots)
                else:
                    rewire_tree(root, None, True, new_roots)

        # create new dataframe that cloned nodes
        new_dataframe = self.dataframe.copy()
        new_dataframe["node"] = new_dataframe["node"].apply(lambda x: node_clone[x])
        new_dataframe.reset_index(level="node", inplace=True, drop=True)

        # create dict that stores aggregation function for each column
        agg_dict = {}
        for col in new_dataframe.columns.tolist():
            if col in self.exc_metrics + self.inc_metrics:
                agg_dict[col] = numpy.sum
            else:
                agg_dict[col] = lambda x: x.iloc[0]

        # perform a groupby to merge nodes with the same callpath
        index_names = self.dataframe.index.names
        agg_df = new_dataframe.groupby(index_names).agg(agg_dict)

        new_graphframe = GraphFrame(Graph(new_roots), agg_df)
        new_graphframe.exc_metrics = self.exc_metrics
        new_graphframe.inc_metrics = self.inc_metrics
        new_graphframe.update_inclusive_columns()

        return new_graphframe

    def unify(self, other):
        """Returns a unified graphframe.

        Ensure self and other have the same graph and same node IDs. This may
        change the node IDs in the dataframe.

        Update the graphs in the graphframe if they differ.
        """
        if self.graph is other.graph:
            return

        node_map = {}
        union_graph = self.graph.union(other.graph, node_map)

        self.dataframe["node"] = self.dataframe["node"].apply(lambda x: node_map[x])
        self.dataframe.set_index(self.dataframe.index.names, inplace=True, drop=False)

        other.dataframe["node"] = other.dataframe["node"].apply(lambda x: node_map[x])
        other.dataframe.set_index(other.dataframe.index.names, inplace=True, drop=False)

        self.graph = union_graph
        other.graph = union_graph

    def _operator(self, other, op, *args, **kwargs):
        """Generic function to apply operator to two dataframes and store
        result in self.

        Arguments:
            self (graphframe): self's graphframe
            other (graphframe): other's graphframe
            op (operator): pandas arithmetic operator

        Return:
            (GraphFrame): self's graphframe modified
        """
        # unioned set of self and other exclusive and inclusive metrics
        all_metrics = list(
            set().union(
                self.exc_metrics, self.inc_metrics, other.exc_metrics, other.inc_metrics
            )
        )

        self.dataframe.update(op(other.dataframe[all_metrics], *args, **kwargs))

        return self

    def add(self, other, *args, **kwargs):
        """Returns the column-wise sum of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.add, *args, **kwargs)

    def sub(self, other, *args, **kwargs):
        """Returns the column-wise difference of two graphframes as a new
        graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.sub, *args, **kwargs)

    def __iadd__(self, other):
        """Computes column-wise sum of two graphframes and stores the result in
        self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and copy of other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.add)

    def __add__(self, other):
        """Returns the column-wise sum of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.add(other)

    def __isub__(self, other):
        """Computes column-wise difference of two graphframes and stores the
        result in self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.sub)

    def __sub__(self, other):
        """Returns the column-wise difference of two graphframes as a new
        graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.sub(other)
